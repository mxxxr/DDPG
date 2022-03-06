import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

class ANet(nn.Module):
    def __init__(self, a_dim, o_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(o_dim, 300)
        self.fc2 = nn.Linear(300, a_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x
    
class CNet(nn.Module):
    def __init__(self, a_dim, o_dim):
        super(CNet, self).__init__()
        self.fc1 = nn.Linear(a_dim + o_dim, 300)
        self.fc2 = nn.Linear(300, 400)
        self.fc3 = nn.Linear(400, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPG():
    def __init__(self, action_dim, observation_dim):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.memory_size = 5000
        # self.memory_pointer = 0
        self.memory = np.zeros((self.memory_size, self.observation_dim * 2 + self.action_dim + 1), dtype=np.float32)
        self.sample_size = 64
        self.gamma = 0.99
        self.learning_rate_A = 0.001
        self.learning_rate_C = 0.0001
        self.soft_para_A = 0.001
        self.soft_para_C = 0.001

        self._build_networks(self.action_dim, self.observation_dim)
        self.criterion = nn.MSELoss()
        self.C_optimizer = optim.Adam(self.C_eval_net.parameters(), lr=self.learning_rate_C)
        self.A_optimizer = optim.Adam(self.A_eval_net.parameters(), lr=self.learning_rate_A)


    def _build_networks(self, a_dim, o_dim):
        self.A_eval_net = ANet(a_dim, o_dim)
        self.A_tar_net = deepcopy(self.A_eval_net)
        self.C_eval_net = CNet(a_dim, o_dim)
        self.C_tar_net = deepcopy(self.C_eval_net)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, r, s_))
        # print(transition)
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # def store_transition(self, s, a, r, s_):
    #     print(self.memory)
    #     transition = np.hstack((s, a, r, s_))
    #     self.memory[self.memory_pointer, :] = transition
    #     self.memory_pointer = 0 if self.memory_pointer >= self.memory_size - 1 else self.memory_pointer + 1

    def choose_action(self, observation):
        return self.A_tar_net(torch.from_numpy(observation).float()).detach().numpy()

    def _sample_batch(self):
        indices = np.random.choice(self.memory_size, size=self.sample_size)
        batch = self.memory[indices, :]
        return batch
    
    def learn(self):
        batch = self._sample_batch()

        temp_s = torch.FloatTensor(batch[:, :self.observation_dim])
        temp_a = torch.FloatTensor(batch[:, self.observation_dim: self.observation_dim + self.action_dim])
        temp_r = torch.FloatTensor(batch[:, self.observation_dim + self.action_dim: self.observation_dim + self.action_dim + 1])
        temp_s_ = torch.FloatTensor(batch[:, -self.observation_dim:])

        ######## Update Critic ########
        action1 = self.A_tar_net(temp_s_).detach()
        Q = self.C_tar_net(torch.cat([temp_s_, action1], 1)).detach()
        y = temp_r + self.gamma * Q

        yy = self.C_eval_net(torch.cat([temp_s, temp_a], 1))
        loss1 = self.criterion(y, yy)
        # print(loss1)

        self.C_optimizer.zero_grad()
        loss1.backward()
        self.C_optimizer.step()

        ######## Update Actor ########
        action2 = self.A_eval_net(temp_s)
        q = self.C_eval_net(torch.cat([temp_s, action2], 1))
        loss2 = torch.mean(-q)
        # print(loss2)

        self.A_optimizer.zero_grad()
        loss2.backward()
        self.A_optimizer.step()

        ######## Soft Replace ########
        for eval_param, target_param in zip(self.A_eval_net.parameters(), self.A_tar_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_para_A) + eval_param.data * self.soft_para_A)
        for eval_param, target_param in zip(self.C_eval_net.parameters(), self.C_tar_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_para_C) + eval_param.data * self.soft_para_C)

        


