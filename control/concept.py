import torch
from NeuralNetwork import basic_nn
import numpy as np
from control import BASE
from torch import nn
from utils import converter
import copy
# state 2
# skill = 8
# action = 2
# skill action = 16
# policy = 2 -> 256 -> 2
# queue = (2 + 2) -> (256) -> 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# s_l = 2 x and y
# a_l = 2 up or down


def state_converter(state):
    x = torch.arange(17).to(DEVICE)
    new_state = torch.zeros(18).to(DEVICE)
    out = torch.exp(-torch.square(x - state[0]))

    new_state[:17] = out
    new_state[-1] = state[-1]
    return new_state


def batch_state_converter(state):
    # print("state", state)
    # print(state.size())
    x = torch.arange(17).to(DEVICE)
    new_state = torch.zeros((len(state), 18)).to(DEVICE)
    out = torch.exp(-torch.square(x.unsqueeze(0) - state[:, 0].squeeze().unsqueeze(-1)))
    # print("out", out.size())
    # print(new_state[:, :17].size())
    new_state[:, :17] = out
    new_state[:, -1] = state[:, -1]
    # print("state", new_state.size())
    return new_state


class Concept(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "concept"
        self.policy_name = "SAC_conti"

        self.policy_list = []
        self.naf_list = []
        self.upd_queue_list = []
        self.base_queue_list = []
        self.naf_out_l = 1
        self.upd_policy = basic_nn.ValueNN(self.s_l, 256, self.a_l).to(self.device)
        self.upd_queue = basic_nn.ValueNN((self.s_l + self.naf_out_l), 256, 1).to(self.device)
        self.base_queue = basic_nn.ValueNN((self.s_l + self.naf_out_l), 256, 1).to(self.device)

        network_p = []
        lr_p = []
        weight_decay_p = []

        network_q = []
        lr_q = []
        weight_decay_q = []

        i = 0
        while i < self.sk_n:

            tmp_policy = copy.deepcopy(self.upd_policy)

            assert tmp_policy is not self.upd_policy, "copy error"
            for name, param in tmp_policy.named_parameters():

                torch.nn.init.uniform_(param, -0.1, 0.1)
                param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
                network_p.append(param)
                if name == "Linear_1.bias":
                    lr_p.append(self.l_r*10)
                else:
                    lr_p.append(self.l_r)
                weight_decay_p.append(0.1)
            self.policy_list.append(tmp_policy)

            tmp_queue = copy.deepcopy(self.upd_queue)
            assert tmp_queue is not self.upd_queue, "copy error"

            for name, param in tmp_queue.named_parameters():
                torch.nn.init.uniform_(param, -0.2, 0.2)
                param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
                network_q.append(param)
                if name == "Linear_1.bias":
                    lr_p.append(self.l_r*10)
                else:
                    lr_p.append(self.l_r)
                lr_q.append(self.l_r)
                weight_decay_q.append(0.1)
            self.upd_queue_list.append(tmp_queue)

            tmp_naf_policy = converter.NAFPolicy(self.s_l, self.a_l, tmp_policy)
            self.naf_list.append(tmp_naf_policy)

            tmp_base_queue = copy.deepcopy(self.base_queue)
            self.base_queue_list.append(tmp_base_queue)
            i = i + 1
        print("assertion")
        assert self.naf_list[0].policy is self.policy_list[0], "assertion error"

        self.optimizer_p = torch.optim.SGD([{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in
                                            zip(network_p, lr_p, weight_decay_p)])

        self.optimizer_q = torch.optim.SGD([{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in
                                            zip(network_q, lr_q, weight_decay_q)])

        self.criterion = nn.MSELoss(reduction='mean')

    def cal_reward(self, t_p_s, t_s, sk_idx):

        t_p_s = batch_state_converter(t_p_s)

        t_s = batch_state_converter(t_s)

        distance_mat = torch.square(t_p_s[:, -1].unsqueeze(0) - t_s[:, -1].unsqueeze(1))

        traj = len(t_p_s) / self.sk_n
        subtract = torch.zeros(len(t_p_s)).to(DEVICE)
        # print(traj)

        i = 0
        while i < len(t_p_s):
            # print("i = ", i)
            if int(sk_idx[i] * traj + i + 1) == int((sk_idx[i] + 1) * traj):
                subtract[i] = torch.tensor(0)
                # print(0)
            else:
                subtract[i] = torch.sum(distance_mat[i][int(sk_idx[i] * traj + i + 1):int((sk_idx[i] + 1) * traj)], -1)
                # print(distance_mat[i][int(sk_idx[i] * traj + i + 1):int((sk_idx[i] + 1) * traj)])
            i = i + 1
        # 0 부터 subtract, ts가 tps랑 같아지는 시점부터
        distance = torch.sum(distance_mat, -1)
        # print(distance)
        reward = distance - subtract
        # print(reward)
        distance_mat = torch.square(t_p_s[:, -1].unsqueeze(0) - t_p_s[:, -1].unsqueeze(1))
        # print("sec distance", distance_mat)
        traj = len(t_p_s) / self.sk_n
        subtract = torch.zeros(len(t_p_s)).to(DEVICE)
        i = 0
        while i < len(t_p_s):
            # print("sec i = ", i)
            if int(sk_idx[i] * traj + i + 1) == int((sk_idx[i] + 1) * traj):
                subtract[i] = torch.tensor(0)
                # print(0)
            else:
                subtract[i] = torch.sum(distance_mat[i][int(sk_idx[i] * traj + i + 1):int((sk_idx[i] + 1) * traj)], -1)
                # print(distance_mat[i][int(sk_idx[i] * traj + i + 1):int((sk_idx[i] + 1) * traj)])
            i = i + 1
        # 0바로 다음부터 subtract, tps랑 target tps랑 같아지는 시점 이후부터 뭐0 부터해도 상관없음
        distance = torch.sum(distance_mat, -1)
        # print(distance)
        reward = reward - distance + subtract
        # print("reward = ",reward)
        """
        narrow_bias_1 = torch.ones(len(t_p_s)).to(DEVICE) * 0.9
        bias1 = torch.sum(torch.square(narrow_bias_1.unsqueeze(0) - t_s[:, 1].unsqueeze(1)), -1)

        narrow_bias_2 = -torch.ones(len(t_p_s)).to(DEVICE) * 0.9
        bias2 = torch.sum(torch.square(narrow_bias_2.unsqueeze(0) - t_s[:, 1].unsqueeze(1)), -1)

        bias = bias1 + bias2
        # print("first bias = ", bias)
        narrow_bias_1 = torch.ones(len(t_p_s)).to(DEVICE) * 0.9
        bias1 = torch.sum(torch.square(narrow_bias_1.unsqueeze(0) - t_p_s[:, 1].unsqueeze(1)), -1)

        narrow_bias_2 = -torch.ones(len(t_p_s)).to(DEVICE) * 0.9
        bias2 = torch.sum(torch.square(narrow_bias_2.unsqueeze(0) - t_p_s[:, 1].unsqueeze(1)), -1)
        # print("sec bias = ", bias1 + bias2)
        bias = bias - bias1 - bias2
        # print("bias", bias[-200:-1])
        reward = reward + bias
        # print("final value = ", reward)
        """
        return reward

    def reward(self,  *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
        t_s = torch.from_numpy(n_s).to(self.device).type(torch.float32)
        return self.cal_reward(t_p_s, t_s, sk_idx)

    def get_performance(self):
        return self.buffer.get_performance()

    def simulate(self, index=None, total=None, pretrain=1, traj=None):
        policy = self.naf_list
        self.buffer.simulate(self.policy.action, policy, self.reward, index, tot_idx=total,
                             pretrain=pretrain, traj_l=traj, encoder=None)

    def update(self, memory_iter, skill_idx, traj_l):
        i = 0
        loss1 = None
        loss2_ary = None
        self.simulate(index=None, total=skill_idx, pretrain=1, traj=traj_l)
        print("iter start")
        while i < memory_iter:
            i = i + 1

            loss2_ary = self.policy.update(self.buffer.get_dataset(), policy_list=self.policy_list,
                                           reward=self.cal_reward,
                                           naf_list=self.naf_list,
                                           upd_queue_list=self.upd_queue_list, base_queue_list=self.base_queue_list,
                                           optimizer_p=self.optimizer_p, optimizer_q=self.optimizer_q,
                                           memory_iter=1, encoder=None)

        loss_ary = loss2_ary
        return loss_ary, self.naf_list

    def load_model(self, path):

        i = 0
        while i < len(self.policy_list):
            self.policy_list[i].load_state_dict(torch.load(path + "/" + self.policy_name + "/" + "policy" + str(i)))
            self.upd_queue_list[i].load_state_dict(torch.load(path + "/" + self.policy_name + "/" + "queue" + str(i)))
            i = i + 1

    def save_model(self, path):

        i = 0
        while i < len(self.policy_list):
            torch.save(self.policy_list[i].state_dict(), path + "/" + self.policy_name + "/" + "policy" + str(i))
            torch.save(self.upd_queue_list[i].state_dict(), path + "/" + self.policy_name + "/" + "queue" + str(i))
            i = i + 1

