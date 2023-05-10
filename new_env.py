import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import registry, register


import sys
import numpy as np
from contextlib import closing
from io import StringIO

"""
game_settings = named tuple which contains variety of settings like game size and rewards
event_ids = named tuple which contains all possible events

You can access with
print(game_settings.rows)
print(event_ids.MOVED_UP)
"""
from settings import game_settings, event_ids
from game_objects import *



# ACTIONS
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
BOMB = 4
WAIT = 5

# STATES
WALL = -1
EXPLOSION = -3
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3
RENDER_CORNERS = False
RENDER_HISTORY = True





"""
Functions needed to train agents:
env.reset() - resets env, returns initial state
env.step(action) - returns next_state, reward, done, log
env._get_obs() - get current state
"""
class BombermanEnv(gym.Env):
    def __init__(self, bombermanrlSettings=game_settings):
        self.screen_height = bombermanrlSettings.rows
        self.screen_width = bombermanrlSettings.cols
        
        # see action space above
        self.action_space = spaces.Discrete(6)

        # NEED TO CHANGE THIS
        self.observation_space = spaces.Box(
            low=-3, high=3, shape=(4+ RENDER_CORNERS+ RENDER_HISTORY, 4), dtype=np.int8
            )
        self.seed()
        self.logger = Log()

        # Start the first game
        self.reset()
        self.env = self

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    

    # resets env, returns initial state
    def reset():
        self.round =0
        self.generate_arena()
        self.player = Agent(1,[1,1])
        self.bombs = []
        self.explosions =[]
        return self._get_obs()
    

    # returns next_state, reward, done, log
    def step(action):
        pass


    # get current state
    def _get_obs():
        pass
