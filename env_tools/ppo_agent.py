from pathlib import Path
import argparse
from collections import namedtuple

import gym
import torch
import torch.nn as nn
import pickle

from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.configs.command_line import cfg_from_cmd
# from easyrl.engine.ppo_engine import PPOEngine
from mod_ppo_engine import ModPPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env

import new_env as bomberman_env
from models import ActorModel
from settings import settings, game_settings

def set_configs(exp_name='ppo_base'):
    set_config('ppo')
    cfg.alg.num_envs = 1
    cfg.alg.episode_steps = 150
    cfg.alg.max_steps = 15000
    cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.alg.env_name = 'Bomberman-v1'
    cfg.alg.save_dir = Path.cwd().absolute().joinpath('data').as_posix()
    cfg.alg.save_dir += f'/{exp_name}'
    setattr(cfg.alg, 'diff_cfg', dict(save_dir=cfg.alg.save_dir))

    print(f'====================================')
    print(f'      Device:{cfg.alg.device}')
    print(f'====================================')

def construct_agent(actor_body, critic_body, env):
    actor = CategoricalPolicy(actor_body,
                                in_features=64,
                                action_dim=env.action_space.n)

    critic = ValueNet(critic_body, in_features=64)
    return PPOAgent(actor=actor, critic=critic, env=env)

def train_ppo(game_settings=game_settings, out_file="ppo"):
    set_configs(out_file)

    set_random_seed(cfg.alg.seed)
    env = make_vec_env(cfg.alg.env_name,
                       cfg.alg.num_envs,
                       seed=cfg.alg.seed)
    env.reset()
    ob_size = env.observation_space.shape[0]

    act_size = env.action_space.n
    actor_body = ActorModel(act_size)
    critic_body = ActorModel(act_size)

    agent = construct_agent(actor_body, critic_body, env)
    # print(agent.actor.body, agent.critic.body)
    runner = EpisodicRunner(agent=agent, env=env)
    engine = ModPPOEngine(agent=agent,
                       runner=runner)
    if not cfg.alg.test:
        engine.train()
    else:
        stat_info, raw_traj_info = engine.eval(render=cfg.alg.render,
                                               save_eval_traj=cfg.alg.save_test_traj,
                                               eval_num=cfg.alg.test_num,
                                               sleep_time=0.04)
        import pprint
        pprint.pprint(stat_info)
    torch.save(agent.actor.body.state_dict(), f'saved_runs/actors/{out_file}.pth')
    torch.save(agent.critic.body.state_dict(), f'saved_runs/critics/{out_file}.pth')
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str)
    # for key in settings:
    #     if type(settings[key]) in [int, bool]:
    #         parser.add_argument(f'--{key}', default=settings[key], type=type(settings[key]))
    args = parser.parse_args()
    # new_settings = dict()
    # for key in settings:
    #     if key in vars(args).keys():
    #         new_settings[key] = getattr(args, key)
    #     else:
    #         new_settings[key] = settings[key]
    # new_game_settings = namedtuple("Settings", new_settings.keys())(*new_settings.values())
    train_ppo(out_file=args.savedir)