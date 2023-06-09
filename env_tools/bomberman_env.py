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

class BombermanEnv(gym.Env):
    def __init__(self, bombermanrlSettings=game_settings):
        self.screen_height = bombermanrlSettings.rows
        self.screen_width = bombermanrlSettings.cols
        
        # see action space above
        self.action_space = spaces.Discrete(6)
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

    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + [self.player]:  # TODO Players...
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free

    def step(self, action):
        reward = 0  # 0 # TODO coins collected as reward
        assert self.action_space.contains(action)
        if action == UP and self.tile_is_free(self.player.x, self.player.y - 1):
            self.player.y -= 1
            self.player.events.append(event_ids.MOVED_UP)
        elif action == DOWN and self.tile_is_free(self.player.x, self.player.y + 1):
            self.player.y += 1
            self.player.events.append(event_ids.MOVED_DOWN)
        elif action == LEFT and self.tile_is_free(self.player.x - 1, self.player.y):
            self.player.x -= 1
            self.player.events.append(event_ids.MOVED_LEFT)
        elif action == RIGHT and self.tile_is_free(self.player.x + 1, self.player.y):
            self.player.x += 1
            self.player.events.append(event_ids.MOVED_RIGHT)
        elif action == BOMB and self.player.bombs_left > 0:
            self.logger.info(
                f'player <{self.player.id}> drops bomb at {(self.player.x, self.player.y)}')
            self.bombs.append(self.player.make_bomb())
            self.player.bombs_left -= 1
            self.player.events.append(event_ids.BOMB_DROPPED)
            reward = 10
        elif action == WAIT:
            self.player.events.append(event_ids.WAITED)
        else:
            reward = -10
        # collect coins
        for coin in self.coins:
            if coin.collectable:
                # for a in self.active_agents:
                a = self.player
                if a.x == coin.x and a.y == coin.y:
                    coin.collectable = False
                    coin.collected = True
                    self.logger.info(
                        f'Agent <{a.id}> picked up coin at {(a.x, a.y)} and receives 1 point')
                    a.update_score(game_settings.reward_coin)
                    a.events.append(event_ids.COIN_COLLECTED)
                    reward = 10
                    # a.trophies.append(Agent.coin_trophy)
        # simulate bombs and explosion
        # bombs
        for bomb in self.bombs:
            # Explode when timer is finished
            if bomb.timer <= 0:
                self.logger.info(
                    f'Agent <{bomb.owner.id}>\'s bomb at {(bomb.x, bomb.y)} explodes')
                blast_coords = bomb.get_blast_coords(self.arena)
                # Clear crates
                for (x, y) in blast_coords:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        # bomb.owner.events.append(e.CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:  # possible bug in GAME engine?==> relive coin
                            if (c.x, c.y) == (x, y):
                                c.collectable = True
                                self.logger.info(f'Coin found at {(x,y)}')
                                bomb.owner.events.append(event_ids.COIN_FOUND)
                # Create explosion
                self.explosions.append(Explosion(blast_coords, bomb.owner))
                # reward= reward+1
                bomb.active = False
                bomb.owner.bombs_left += 1
            # Progress countdown
            else:
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]
        # explosions
        # Explosions
        agents_hit = set()
        detonation = False
        for explosion in self.explosions:
            # Kill agents
            if explosion.timer > 1:
                reward += 1
                detonation = True
                # for a in self.active_agents:
                a = self.player
                if self.player.alive:
                    if a.alive and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.add(a)
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(
                                f'Agent <{a.id}> blown up by own bomb')
                            a.events.append(event_ids.KILLED_SELF)
                            # explosion.owner.trophies.append(Agent.suicide_trophy)
                        else:
                            self.logger.info(
                                f'Agent <{a.id}> blown up by agent <{explosion.owner.id}>\'s bomb')
                            self.logger.info(
                                f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(game_settings.reward_kill)
                            explosion.owner.events.append(event_ids.KILLED_OPPONENT)
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False
            # Progress countdown
            explosion.timer -= 1
        a = self.player
        if a in agents_hit:
            a.alive = False
        #    self.active_agents.remove(a)
        #    a.events.append(e.GOT_KILLED)
        #    for aa in self.active_agents:
        #        if aa is not a:
        #            aa.events.append(e.OPPONENT_ELIMINATED)
        #    self.put_down_agent(a)
        self.explosions = [e for e in self.explosions if e.active]
        # check whether coins where collected
        self.round = self.round+1
        done = self.check_if_all_coins_collected(
        ) or self.all_players_dead() or self.round > 200
        #if detonation:
        #    reward= 10
        if self.round > 200:
            reward = -1
        if not self.player.alive:
            reward = -1000
        # reward = reward + self.player.score*10

        return (self._get_obs(), reward, done, {})

    def check_if_all_coins_collected(self):
        return len([c for c in self.coins if c.collected]) == len(self.coins)

    def all_players_dead(self):
        return not self.player.alive
        # return length([a for a in self.players if a])

    """
    Function that returns the viewed image or so called observation map values:
       2: Coin
       -1: WALL
       -2: Bomb
       -3: Explosion
       0 : Free
       1 : Crate
       3,4,5,6: player
    """
    def _get_obs2(self):
        return self._render_4_perspective()

    def _get_obs(self):
        rendered_map = np.copy(self.arena)
        # add coins
        for coin in self.coins:
            if coin.collectable:
                rendered_map[coin.x, coin.y] = 2
        # add bombs
        for bomb in self.bombs:
            rendered_map[bomb.x, bomb.y] = -2
        for explosion in self.explosions:
            for e in explosion.blast_coords:
                rendered_map[e[0], e[1]] = -3
        # TODO add players
        rendered_map[self.player.x, self.player.y] = 3

        return rendered_map

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
    
    def generate_arena(self):
        # Arena with wall and crate layout
        self.arena = (np.random.rand(game_settings.cols, game_settings.rows) < game_settings.crate_density).astype(np.int8)
        self.arena[:1, :] = -1
        self.arena[-1:,:] = -1
        self.arena[:, :1] = -1
        self.arena[:,-1:] = -1
        self.coins = []
        for x in range(game_settings.cols):
            for y in range(game_settings.rows):
                if (x+1)*(y+1) % 2 == 1:
                    self.arena[x,y] = -1
                    self.coins.append(Coin((x,y)))# Adding coins every where
        # Starting positions
        self.start_positions = [(1,1), (1,game_settings.rows-2), (game_settings.cols-2,1), (game_settings.cols-2,game_settings.rows-2)]
        np.random.shuffle(self.start_positions)
        for (x,y) in self.start_positions:
            for (xx,yy) in [(x,y), (x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                if self.arena[xx,yy] == 1:
                    self.arena[xx,yy] = 0
        # Distribute coins evenly
        
        #for i in range(3):
        #    for j in range(3):
        #        n_crates = (self.arena[1+5*i:6+5*i, 1+5*j:6+5*j] == 1).sum()
        #        while True:
        #            x, y = np.random.randint(1+5*i,6+5*i), np.random.randint(1+5*j,6+5*j)
        #            if n_crates == 0 and self.arena[x,y] == 0:
        #                self.coins.append(Coin((x,y)))
        #                self.coins[-1].collectable = True
        #                break
        #            elif self.arena[x,y] == 1:
        #                self.coins.append(Coin((x,y)))
        #                break

    def reset(self):
        self.round =0
        self.generate_arena()
        self.player = Agent(1,[1,1])
        self.bombs = []
        self.explosions =[]
        return self._get_obs()
    
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