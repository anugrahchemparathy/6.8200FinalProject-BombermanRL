import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gym
from copy import deepcopy
import bomberman_env

from torchvision.models import resnet18
from torch.autograd import Variable

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 20000
Q_NETWORK_ITERATION = 100

env = gym.make("Bomberman-v1") ### REPLACE WITH YOUR ENVIRONMENT
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class ResNet(nn.Module):
    """docstring for ResNet"""
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, NUM_ACTIONS)

    def forward(self,x):
        x = self.resnet(x)
        return x
    
# create a replay buffer
class CyclicBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.cur_pos = 0

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def append(self, data):
        if self.cur_pos < self.capacity:
            self.buffer.append(data)
            self.cur_pos += 1
        else:
            self.buffer = self.buffer[1:]
            self.buffer.append(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def get_all(self):
        return deepcopy(self.buffer)
    
    def clear(self):
        self.buffer.clear()

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = ResNet(), ResNet()
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.memory = CyclicBuffer(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.HuberLoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append(dict(
            ob=torch.tensor(state, dtype=torch.float32),
            next_ob=torch.tensor(next_state, dtype=torch.float32),
            action=torch.tensor(action),
            reward=torch.tensor(reward),
            done=torch.tensor(done)
        ))
        self.memory_counter += 1


    def learn(self):
        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch = self.memory.sample(self.batch_size)
        batch_state = torch.stack([x["ob"] for x in batch])
        batch_action = torch.stack([x["action"].cuda() for x in batch]).cuda()
        batch_reward = torch.stack([x["reward"] for x in batch]).cuda()
        batch_next_state = torch.stack([x["next_ob"] for x in batch if not x["done"]])
        batch_done = torch.stack([x["done"] for x in batch]).cuda()
        batch_not_done = [not x["done"] for x in batch]

        
        if ob_batch.shape[1] != 3:
            ob_batch = ob_batch.permute(0, 3, 1, 2)
        if next_ob_batch.shape[1] != 3:
            next_ob_batch = next_ob_batch.permute(0, 3, 1, 2)
        

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action.unsqueeze(1))
        q_next = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            q_next[batch_not_done] = self.target_net(batch_next_state).detach().max(1)[0]
        q_target = batch_reward + GAMMA * q_next
        loss = self.loss_func(q_eval, q_target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    dqn = DQN()
    episodes = 400
    print("Collecting Experience....")
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            # env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            dqn.store_transition(state, action, reward, next_state, done)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state

if __name__ == '__main__':
    main()