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
from settings import game_settings, event_ids, rewards
from game_objects import *
from arenas import *

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
    """
    FIELDS
    
    """
    def __init__(self, bombermanrlSettings=game_settings):
        self.screen_height = bombermanrlSettings.rows
        self.screen_width = bombermanrlSettings.cols

        # see action space above
        self.action_space = spaces.Discrete(6)

        # NEED TO CHANGE THIS
        # self.observation_space = spaces.Box(
        #     low=-3, high=3, shape=(4+ RENDER_CORNERS+ RENDER_HISTORY, 4), dtype=np.int8
        #     )
        self.seed()
        self.logger = Log()

        # Start the first game
        self.reset()
        self.env = self

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # gets arena randomly selected from arenas.py
    def generate_arena(self):
        self.arena = get_arena()

        
        

    """
    given the current player and action, update the player's location
    and returns reward based on coins collected
    """
    def update_player_loc(action):
        reward = 0
        
        if action == UP and self.tile_is_free(self.player.x, self.player.y - 1):
            self.player.y -= 1
            self.player.events.append(event_ids.MOVED_UP)
            reward += rewards.valid_move
        
        elif action == DOWN and self.tile_is_free(self.player.x, self.player.y + 1):
            self.player.y += 1
            self.player.events.append(event_ids.MOVED_DOWN)
            reward += rewards.valid_move
        
        elif action == LEFT and self.tile_is_free(self.player.x - 1, self.player.y):
            self.player.x -= 1
            self.player.events.append(event_ids.MOVED_LEFT)
            reward += rewards.valid_move
        
        elif action == RIGHT and self.tile_is_free(self.player.x + 1, self.player.y):
            self.player.x += 1
            self.player.events.append(event_ids.MOVED_RIGHT)
            reward += rewards.valid_move
        
        elif action == BOMB and self.player.bombs_left > 0:
            self.logger.info(f'player <{self.player.id}> drops bomb at {(self.player.x, self.player.y)}')
            self.bombs.append(self.player.make_bomb())
            self.player.events.append(event_ids.BOMB_DROPPED)
            reward += rewards.place_bomb
        
        elif action == WAIT:
            self.player.events.append(event_ids.WAITED)
            reward += rewards.wait
        
        else:
            reward += rewards.invalid_action
        
        # collect coins
        for coin in self.coins:
            if coin.collectable:
                a = self.player
                if a.x == coin.x and a.y == coin.y:
                    coin.collectable = False
                    coin.collected = True
                    self.logger.info(f'Agent <{a.id}> picked up coin at {(a.x, a.y)} and receives 1 point')
                    a.update_score(game_settings.reward_coin)
                    a.events.append(event_ids.COIN_COLLECTED)
                    reward += rewards.collect_coin

        return reward

    """
    explodes bomb and modifies state of all crates in blasted area. creates a new explosion.
    """
    def explode_bomb(bomb):
        self.logger.info(f'Agent <{bomb.owner.id}>`s bomb at {(bomb.x, bomb.y)} explodes')
        blast_coords = bomb.get_blast_coords(self.arena)
        # Clear crates
        for (x, y) in blast_coords:
            if self.arena[x, y] == 1:
                self.arena[x, y] = 0

        # Create explosion, no need to reward - agent was rewarded for placing the bomb
        self.explosions.append(Explosion(blast_coords, bomb.owner))
        bomb.active = False
        bomb.owner.bombs_left += 1

    """
    updates explosions (ie timer of explosions) and kills agents that lie in explosion range
    TODO: set reward for agent
    """
    def place_explosions(self):
        detonation = False
        for explosion in self.explosions:
            # Kill agents
            if explosion.timer > 1:
                detonation = True
                a = self.player
                if a.alive:
                    if (a.x, a.y) in explosion.blast_coords:
                        a.alive = False
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(
                                f'Agent <{a.id}> blown up by own bomb')
                            a.events.append(event_ids.KILLED_SELF)
                        else:
                            self.logger.info(f'Agent <{a.id}> blown up by agent <{explosion.owner.id}>\'s bomb')
                            self.logger.info(f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(game_settings.reward_kill)
                            explosion.owner.events.append(event_ids.KILLED_OPPONENT)
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False
            explosion.timer -= 1
          
        self.explosions = [e for e in self.explosions if e.active]

    def all_players_dead(self):
        return not self.player.alive

    """
    ==========================================================
    CORE ENV FUNCTIONS
    ==========================================================
    """
    
    # resets env, returns initial state
    def reset(self):
        self.round = 0
        self.generate_arena()
        self.player = Agent(1, [1, 1])
        self.bombs = []
        self.explosions = []
        return self._get_obs()
    
    # returns next_state, reward, done, log
    def step(self, action):
        reward = 0
        
        # player locations
        reward += self.update_player_loc(action)
        
        # update bombs
        for bomb in self.bombs:
            bomb.timer -= 1
            if bomb.timer == 0:
                self.explode_bomb(bomb)
        self.bombs = [b for b in self.bombs if b.active]

        self.place_explosions()

        self.round = self.round+1
        done = self.check_if_all_coins_collected() or self.all_players_dead() or self.round > 200

        if self.round > 200:
            reward += rewards.game_timeout
        if not self.player.alive:
            reward += rewards.agent_died

        return (self._get_obs(), reward, done, {})


    # get current state
    def _get_obs(self):
        rendered_map = np.copy(self.arena)
        
        # add coins
        for coin in self.coins:
            if coin.collectable:
                rendered_map[coin.x, coin.y] = COIN
        
        # add bombs
        for bomb in self.bombs:
            rendered_map[bomb.x, bomb.y] = BOMB
        for explosion in self.explosions:
            for e in explosion.blast_coords:
                rendered_map[e[0], e[1]] = EXPLOSION
        
        # TODO add players
        rendered_map[self.player.x, self.player.y] = PLAYER

        return rendered_map
