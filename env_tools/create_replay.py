from new_env import *

history = []
actions = [RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN, DOWN, LEFT, LEFT, RIGHT, RIGHT, LEFT, LEFT, WAIT, LEFT, DOWN, DOWN, RIGHT, UP, DOWN, RIGHT, LEFT, RIGHT, BOMB, LEFT, DOWN, DOWN, DOWN, DOWN, DOWN, DOWN, BOMB, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT, DOWN, DOWN, LEFT, DOWN, DOWN, DOWN, LEFT, DOWN, LEFT, DOWN, BOMB, BOMB, LEFT, BOMB, BOMB]



benv = BombermanEnv(game_settings)
history.append(benv.render())

for action in actions:
    benv.step(action)
    history.append(benv.render()) 

# write history to a txt file with newlines between each entry
with open('replay.txt', 'w') as outfile:
    outfile.write('\n'.join(history))