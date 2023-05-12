from new_env import *
from parse_actions import parse_actions

enums = {
    'UP': UP,
    'DOWN': DOWN,
    'LEFT': LEFT,
    'RIGHT': RIGHT,
    'BOMB': BOMB,
    'WAIT': WAIT
}

def generate_replay(actions, out_file):
    benv = BombermanEnv(game_settings)
    benv.reset(np.array([2,15]))
    history = []
    history.append(benv.render())
    for action in actions:
        benv.step(action)
        history.append(benv.render())
    with open(f'replays/{out_file}', 'w') as f:
        for state in history:
            f.write("%s\n" % state)

if __name__ == '__main__':
    actions = parse_actions('file.txt')
    actions = [enums[a] for a in actions]
    generate_replay(actions, 'maze_replay_long.txt')
