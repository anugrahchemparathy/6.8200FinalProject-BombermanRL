import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import gym
from copy import deepcopy
import env_tools.bomberman_env as bomberman_env
from torch import optim
from tqdm import tqdm

from models import ActorModel

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

import gym.utils.seeding as seeding
def get_default_config():
    env = DoorKeyEnv5x5()
    env._np_random, seed_ = seeding.np_random(seed)
    env.action_space.seed(seed)
    env = ImgObsWrapper(env)
    config = dict(
        env=env,
        learning_rate=0.00025,
        gamma=0.99,
        memory_size=200000,
        initial_epsilon=1.0,
        min_epsilon=0.1,
        max_epsilon_decay_steps=100000,
        warmup_steps=500,
        target_update_freq=2000,
        batch_size=32,
        device=None,
        disable_target_net=False,
        enable_double_q=False
    )
    return config
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

class DQNAgent():
    """docstring for DQN"""
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.eval_net, self.target_net = ActorModel(), ActorModel()
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.memory = CyclicBuffer(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.HuberLoss()

    def reset(self):
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.qnet = ActorModel(self.env.action_space.n)
        self.target_qnet = deepcopy(self.qnet)
        self.memory = CyclicBuffer(self.memory_size)
        self.optim = optim.Adam(self.qnet.parameters(), lr=self.learning_rate)
        self.qnet.to(self.device)
        self.target_qnet.to(self.device)

        self.loss_criterion = nn.HuberLoss()
        self.epsilon = self.initial_epsilon
        self.ep_reduction = (self.epsilon - self.min_epsilon) / float(self.max_epsilon_decay_steps)

    @torch.no_grad()
    def get_action(self, ob, greedy_only=False):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(self.device)
        q_val = self.qnet(ob)
        action = self.epsilon_greedy_policy(q_val, greedy_only=greedy_only)
        return action

    def epsilon_greedy_policy(self, q_values, greedy_only=False):
        greedy_act = torch.argmax(q_values).cpu()
        greedy = greedy_only or np.random.uniform() > self.epsilon
        return greedy_act if greedy else self.env.action_space.sample()

    def add_to_memory(self, ob, next_ob, action, reward, done):
        self.memory.append(dict(
            ob=torch.tensor(ob),
            next_ob=torch.tensor(next_ob),
            action=torch.tensor(action),
            reward=torch.tensor(reward),
            done=torch.tensor(done)
        ))
    
    def update_Q(self):
        # we only start updating the Q network if there are enough samples in the replay buffer
        if len(self.memory) < self.warmup_steps:
            return 0

        b = self.memory.sample(self.batch_size)
        batch = dict()
        for key in self.memory[0]:
            batch[key] = torch.stack([self.memory[idx][key] for idx in b]).to(self.device)
            if key in ['action', 'reward', 'done']:
                batch[key] = batch[key].unsqueeze(1)
        q_curr = self.qnet(batch['ob'])
        pred = q_curr.gather(1, batch['action'])
        q_next = self.target_qnet(batch['next_ob'])
        if self.enable_double_q:
            action = torch.argmax(q_curr, 1).reshape(-1, 1)
            maxQ = q_next.gather(1, action)
        else:
            with torch.no_grad():
                maxQ = torch.max(q_next, 1)[0].unsqueeze(1)

        target = batch['reward'] + self.gamma * maxQ * (~batch['done'])
        loss = self.loss_criterion(pred, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()
    
    def decay_epsilon(self):
        if self.epsilon - self.ep_reduction >= self.min_epsilon:
            self.epsilon -= self.ep_reduction

    def update_target_qnet(self, step):
        if step % self.target_update_freq == 0:
            for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
                target_param.data.copy_(
                    param.data
                )

def main():
    config = get_default_config()
    agent = DQNAgent(**config)
    show_progress = False
    max_steps = 100000 
    env = agent.env
    n_runs = 1

    rewards = []
    log = []

    for i in tqdm(range(n_runs), desc='Runs'):
        ep_rewards = []
        ep_steps = []
        agent.reset()
        # we plot the smoothed return values
        smooth_ep_return = deque(maxlen=10)
        ob = env.reset()
        ret = 0
        num_ep = 0
        for t in tqdm(range(max_steps), desc='Step'):
            if len(agent.memory) < agent.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(ob)
            next_ob, reward, done, truncated, info = env.step(action)
            true_done = done and not info.get('TimeLimit.truncated', False)
            agent.add_to_memory(ob, next_ob, action, reward, true_done)
            agent.update_Q()
            ret += reward
            ob = next_ob
            if done or truncated:
                ob = env.reset()
                smooth_ep_return.append(ret)
                ep_rewards.append(np.mean(smooth_ep_return))
                ep_steps.append(t)
                ret = 0
                num_ep += 1
                if show_progress:
                    print(f'Step:{t}  epsilon:{agent.epsilon}  '
                        f'Smoothed Training Return:{np.mean(smooth_ep_return)}')
                if num_ep % 10 == 0:
                    test_ret = self.test()
                    if show_progress:
                        print('==========================')
                        print(f'Step:{t} Testing Return: {test_ret}')
            agent.decay_epsilon()
            agent.update_target_qnet(t)

        rewards.append(ep_rewards)
        run_log = pd.DataFrame({'return': ep_rewards,  
                                'steps': ep_steps,
                                'episode': np.arange(len(ep_rewards)), 
                                'epsilon': agent.initial_epsilon})
        log.append(run_log)
    return log

if __name__ == '__main__':
    main()