import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA = 0.98


class Memory:
    def __init__(self, env, step_size, done_penalty, skill_num, capacity, dataset, dataloader):
        self.env = env
        self.step_size = step_size
        self.done_penalty = done_penalty
        self.performance = 0
        self.skill_num = skill_num
        self.s_l = len(env.observation_space.sample())
        self.a_l = len(env.action_space.sample())
        self.dataset = dataset
        self.dataloader = dataloader
        self.capacity = capacity

    def simulate(self, action_f, policy, reward_f, index=None, tot_idx=None, pretrain=1, traj_l=0, encoder=None):
        total_num = 0
        pause = 0
        failure = 1
        while len(self.dataset) != 0:
            self.dataset.pop()
        if traj_l != 0:
            if index is not None:
                _index = index
                circular = 0
            else:
                _index = 0
                circular = 1
            while total_num < tot_idx*traj_l:
                n_p_s = self.env.reset()
                t = 0
                while t < (tot_idx*traj_l) - total_num: # if pg, gain accumulate
                    with torch.no_grad():
                        n_a = action_f(n_p_s, policy, _index, encoder=encoder)
                    # n_s, n_r, n_i = self.env.step(n_a[_index*self.a_l:(_index+1)*self.a_l])

                    n_s, n_r, n_i = self.env.step(n_a.squeeze())
                    if (t+1) == traj_l:
                        done = 1
                    else:
                        done = 0
                    self.dataset.push(n_p_s, n_a, n_s, t, done, _index)
                    # self.dataset.push(n_p_s, n_a, n_s, n_r, done, _index)
                    # we need index.. so have to convert dataset
                    n_p_s = n_s
                    t = t + 1
                    if t == traj_l:
                        if circular == 1:
                            _index = _index + 1
                        total_num += t
                        failure = failure + 1
                        break
        else:
            while total_num < self.capacity - pause:
                if index is not None:
                    _index = index
                else:
                    _index = np.random.randint(self.skill_num)
                n_p_s = self.env.reset()
                t = 0
                while t < self.capacity - total_num: # if pg, gain accumulate
                    with torch.no_grad():
                        n_a = action_f(n_p_s, _index, encoder=encoder)
                    n_s, n_r, n_d, n_i = self.env.step(n_a)
                    self.dataset.push(n_p_s, n_a, n_s, n_r, np.float32(n_d), _index)
                    # we need index.. so have to convert dataset
                    n_p_s = n_s
                    t = t + 1
                    if n_d:
                        total_num += t
                        t = 0
                        failure = failure + 1
                        break
                pause = t
        if pretrain == 1:
            with torch.no_grad():
                print("in buffer")
                reward = reward_f(next(iter(self.dataloader)))
                # print("reward = ", reward[-100:-1])
                print("in buffer f")
            self.reward_converter(reward)
        else:
            pass

        pre_observation, action, observation, reward, done, skill_idx = next(iter(self.dataloader))

        total_performance = np.sum(reward)
        self.performance = total_performance / failure
        self._reward_converter()
        return self.performance

    def reward_converter(self, _reward):
        pre_observation, action, observation, reward, done, skill_idx = next(iter(self.dataloader))
        reward = _reward.cpu().numpy()
        global_index = 0
        while global_index < len(done):
            self.dataset.push(pre_observation[global_index], action[global_index], observation[global_index],
                              reward[global_index], np.float32(done[global_index]), skill_idx[global_index])
            global_index += 1
        return self.dataset

    def get_performance(self):
        return self.performance

    def get_dataset(self):
        return next(iter(self.dataloader))

    def _reward_converter(self):
        print("now converted")
        t = 0
        pre_observation, action, observation, reward, done, skill_idx = next(iter(self.dataloader))
        # cal per trajectary to_end length ex) 4 3 2 1 6 5 4 3 2 1
        # set step to upper bound ex) step = 5 ->  4 3 2 1 5 5 4 3 2 1
        global_index = len(done) - 1
        local_index = 0
        while 0 <= global_index:
            if done[global_index] == 1:
                local_index = 1
                done[global_index] = local_index
                reward[global_index] -= self.done_penalty
            else:
                local_index = local_index + 1
                if local_index > self.step_size:
                    local_index = self.step_size
                done[global_index] = local_index
            global_index = global_index - 1
        # cal newreward per state-action pair

        global_index = 0
        while global_index < len(done):
            observation[global_index] = observation[int(global_index + done[global_index] - 1)]
            # change observation to last step indexed observation state
            local_index = 1
            while local_index < done[global_index]:
                tmp = reward[global_index + local_index] * GAMMA ** local_index
                reward[global_index] += tmp
                local_index = local_index + 1
            global_index += 1
        global_index = 0
        while global_index < len(done):
            self.dataset.push(pre_observation[global_index], action[global_index], observation[global_index],
                              reward[global_index], np.float32(done[global_index]), skill_idx[global_index])
            global_index += 1
        return self.dataset

