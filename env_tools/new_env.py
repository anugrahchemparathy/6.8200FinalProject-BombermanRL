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
        self.observation_space = spaces.Box(
            low=-3, high=3, shape=(17, 17), dtype=np.int8
            )
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
    def update_player_loc(self, action):
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
    def explode_bomb(self, bomb):
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
        self.coins = []
        return self._get_obs()
    
    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + [self.player]:  # TODO Players...
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free
    
    def check_if_all_coins_collected(self):
        return len([c for c in self.coins if c.collected]) == len(self.coins)

    def all_players_dead(self):
        return not self.player.alive
    
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

    def render(self,history,mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        # 2: Coin
        #    -1: WALL
        #    -2: Bomb
        #    -3: Explosion
        #    0 : Free
        #    1 : Crate
        #    3,4,5,6: player
        map = self._get_obs()
        st = ""
        outfile.write("\n")
        for zeile in map:
            for element in zeile:
                outfile.write("{}".format(["💥","💣","❌","👣","❎","🏆","😎"][element+3]))
                st += "{}".format(["💥","💣","❌","👣","❎","🏆","😎"][element+3])
            outfile.write("\n")
            st += "\n"
        view = self._get_obs2()
        history.append(st)
        if not RENDER_HISTORY:
            for zeile in view:
                for element in zeile:
                    outfile.write("{}".format(["💥","💣","❌","👣","❎","🏆","😎"][element+3]))
                outfile.write("\n")
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

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

    def _get_obs2(self):
        return self._render_4_perspective()

    def _render_4_perspective(self, distance=4):
        result = np.zeros((4+RENDER_CORNERS+ RENDER_HISTORY, distance),dtype=np.int8)
        x = self.player.x
        y = self.player.y
        k = 0
        for it_x, it_y in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            wand = False
            for i in range(distance):  # should we be able to look over walls? --> currently not
                if(wand):
                    result[k, i] = WALL
                else:
                    # TODO; Wand bedingung updaten
                    if x+it_x*(i+1) < 0 or 0 > y+it_y*(i+1) or x+it_x*(i+1) > game_settings.cols or game_settings.rows < y+it_y*(i+1):
                        wand = True
                        result[k, i] = WALL
                    elif self.arena[x+it_x*(i+1), y+it_y*(i+1)] == WALL:
                        wand = True
                        result[k, i] = WALL
                    else:
                        result[k,i] = self.arena[x+it_x*(i+1), y+it_y*(i+1)] # forgotten first important!
                        for b in self.bombs:
                            if b.x == x+it_x*(i+1) and b.y == y+it_y*(i+1):
                                result[k, i] = -2
                        for c in self.coins:
                            if c.x == x+it_x*(i+1) and c.y == y+it_y*(i+1) and c.collectable:
                                result[k, i] = COIN  # TODO Players, Explosions
                        for e in self.explosions:
                            if (x+it_x*(i+1), y+it_y*(i+1)) in e.blast_coords:
                                result[k,i] = EXPLOSION
            k = k+1
        k= distance
        if RENDER_CORNERS:
            i =0 #adding corners
            for it_x, it_y in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
                # TODO; Wand bedingung updaten
                if x+it_x < 0 or 0 > y+it_y or x+it_x > game_settings.cols or game_settings.rows < y+it_y:
                    wand = True
                    result[k, i] = WALL
                elif self.arena[x+it_x,y+it_y] == WALL:
                    wand= True
                    result[k,i]= WALL
                else:
                    result[k,i] = self.arena[x+it_x, y+it_y] # forgotten first important!
                    for b in self.bombs:
                        if b.x == x+it_x and b.y == y+it_y:
                            result[k,i] = -2
                    for c in self.coins:
                        if c.x == x+it_x and c.y == y+it_y and c.collectable:
                            result[k,i] = COIN
                    for e in self.explosions:
                            if (x+it_x, y+it_y) in e.blast_coords:
                                result[k,i] = EXPLOSION
                i = i+1
            k = k+1 # inc by one 
        if RENDER_HISTORY:
            for i in range(distance):
                if len(self.player.events)<=i:
                    result[k,i]=-1
                else:
                    result[k,i]=self.player.events[len(self.player.events)-i-1]
        return result#.reshape(4*distance)

module_name = __name__
env_name = 'Bomberman-v1'
if env_name in registry:
    del registry[env_name]
register(
    id=env_name,
    entry_point=f'{module_name}:BombermanEnv',
)

if __name__ == "__main__":
    history = []
    benv = BombermanEnv(game_settings)
    benv.step(RIGHT)
    benv.step(BOMB)
    benv.render(history)
    benv.step(LEFT)
    benv.render(history)
    benv.step(WAIT)
    benv.render(history)
    benv.step(WAIT)
    benv.render(history)
    benv.step(WAIT)
    benv.render(history)
    benv.step(WAIT)
    benv.render(history)
    with open('states.txt', 'w') as f:
        for state in history:
            f.write("%s\n" % state)