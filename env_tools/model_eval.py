import numpy as np
from tqdm.notebook import tqdm
from torch import nn
import torch
import gym
from easyrl.agents.base_agent import BaseAgent
from easyrl.utils.torch_util import DictDataset
from easyrl.utils.torch_util import load_state_dict
from easyrl.utils.torch_util import load_torch_model
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.torch_util import save_model
from easyrl.utils.torch_util import freeze_model
from easyrl.utils.common import save_traj

def create_actor(env):
    ob_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor_body = 'pass' ## REPLACE
    actor = 'pass' ## REPLACE
    return actor

def load_agent(env, device, expert_model_path='pusher_expert_model.pt'):
    expert_actor = create_actor(env=env)
    expert_agent = BasicAgent(actor=expert_actor) ##REPLACE WITH YOUR AGENT
    print(f'Loading expert model from: {expert_model_path}.')
    ckpt_data = torch.load(expert_model_path, map_location=torch.device(f'{device}'))
    load_state_dict(expert_agent.actor,
                    ckpt_data['actor_state_dict'])
    freeze_model(expert_agent.actor)


def run_inference(agent, env, num_trials, return_on_done=False, sample=True, disable_tqdm=False, render=False):
    runner = EpisodicRunner(agent=agent, env=env)
    trajs = []
    for trial_id in tqdm(range(num_trials), desc='Run', disable=disable_tqdm):
        env.reset()
        traj = runner(time_steps=cfg.alg.episode_steps,
                      sample=sample,
                      return_on_done=return_on_done,
                      evaluation=True,
                      render_image=render)
        trajs.append(traj)
    return trajs

def eval_agent(agent, env, num_trials, disable_tqdm=False, render=False):
    trajs = run_inference(agent, env, num_trials, return_on_done=True, 
                          disable_tqdm=disable_tqdm, render=render)
    tsps = []
    successes = []
    rets = []
    for traj in trajs:
        tsps = traj.steps_til_done.copy().tolist()
        rewards = traj.raw_rewards
        infos = traj.infos
        for ej in range(rewards.shape[1]):
            ret = np.sum(rewards[:tsps[ej], ej])
            rets.append(ret)
            successes.append(infos[tsps[ej] - 1][ej]['success'])
        if render:
            save_traj(traj, 'tmp')
    ret_mean = np.mean(rets)
    ret_std = np.std(rets)
    success_rate = np.mean(successes)
    return success_rate, ret_mean, ret_std, rets, successes