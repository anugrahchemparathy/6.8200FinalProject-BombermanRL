from pathlib import Path

import gym
import torch
import torch.nn as nn

from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env

import env_tools.bomberman_env as bomberman_env
from env_tools.models import ResNet

def set_configs(exp_name='bc'):
    set_config('ppo')
    cfg.alg.num_envs = 1
    cfg.alg.episode_steps = 150
    cfg.alg.max_steps = 600000
    cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.alg.env_name = 'Bomberman-v1'
    cfg.alg.save_dir = Path.cwd().absolute().joinpath('data').as_posix()
    cfg.alg.save_dir += f'/{exp_name}'
    setattr(cfg.alg, 'diff_cfg', dict(save_dir=cfg.alg.save_dir))

    print(f'====================================')
    print(f'      Device:{cfg.alg.device}')
    print(f'====================================')

def main():
    set_configs()

    set_random_seed(cfg.alg.seed)
    env = make_vec_env(cfg.alg.env_name,
                       cfg.alg.num_envs,
                       seed=cfg.alg.seed)
    env.reset()
    ob_size = env.observation_space.shape[0]

    # actor_body = MLP(input_size=ob_size,
    #                  hidden_sizes=[64],
    #                  output_size=64,
    #                  hidden_act=nn.ReLU,
    #                  output_act=nn.ReLU)
    actor_body = ResNet()
    critic_body = ResNet()

    # critic_body = MLP(input_size=ob_size,
    #                   hidden_sizes=[64],
    #                   output_size=64,
    #                   hidden_act=nn.ReLU,
    #                   output_act=nn.ReLU)
    act_size = env.action_space.n
    actor = CategoricalPolicy(actor_body,
                                in_features=64,
                                action_dim=act_size)

    critic = ValueNet(critic_body, in_features=64)
    agent = PPOAgent(actor=actor, critic=critic, env=env)
    runner = EpisodicRunner(agent=agent, env=env)
    engine = PPOEngine(agent=agent,
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
    env.close()


if __name__ == '__main__':
    main()