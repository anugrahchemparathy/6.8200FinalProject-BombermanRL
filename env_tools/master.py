from new_env import BombermanEnv

env = BombermanEnv()

# STATES
WALL = -1
EXPLOSION = -3
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3

# ACTIONS
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
BOMB = 4
WAIT = 5

dirs = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0)
}

def bfs_to_coin(ob):
    playerX, playerY = None, None
    for i in range(17):
        for j in range(17):
            if ob[0][i][j] == PLAYER:
                playerX, playerY = i, j

    print("player", playerX, playerY)
    q = [(playerX, playerY, [])]
    ind = 0
    visited = set()

    while ind < len(q):
        curX, curY, path = q[ind]

        if ob[0][curX][curY] == COIN:
            return path
        
        if (curX, curY) in visited:
            ind+=1
            continue
        else:
            visited.add((curX, curY))

        for (key, (dirX, dirY)) in dirs.items():
            if 0<= curX + dirX < 17 and 0<= curY + dirY < 17:
                val = ob[0][curX+dirX][curY+dirY]
                if  val == FREE or val == COIN:
                    q.append((curX+dirX, curY+dirY, path+[key]))
    
        ind+=1

    return []

power = 1
WAIT_TIME = 4
def get_blast_coords(x, y, arena):
        blast_coords = [(x, y)]

        for i in range(1, power+1):
            if arena[x+i, y] == -1: break
            blast_coords.append((x+i, y))
        for i in range(1, power+1):
            if arena[x-i, y] == -1: break
            blast_coords.append((x-i, y))
        for i in range(1, power+1):
            if arena[x, y+i] == -1: break
            blast_coords.append((x, y+i))
        for i in range(1, power+1):
            if arena[x, y-i] == -1: break
            blast_coords.append((x, y-i))

        return blast_coords

def place_bomb_and_dodge(ob, playerX, playerY):
    actions = [BOMB]
    blast = set(get_blast_coords(playerX, playerY, ob[0]))
    print("Bomb", playerX, playerY)
    print(blast)

    q = [(playerX, playerY, [])]
    dodge_path = None
    visited = set()
    ind = 0
    while ind < len(q):
        curX, curY, path = q[ind]

        if not ((curX, curY) in blast):
            print("Dodged")
            dodge_path = path
            break
        
        if (curX, curY) in visited:
            ind+=1
            continue
        else:
            visited.add((curX, curY))

        for (key, (dirX, dirY)) in dirs.items():
            if 0<= curX + dirX < 17 and 0<= curY + dirY < 17:
                val = ob[0][curX+dirX][curY+dirY]
                if  val == FREE or val == COIN or val == PLAYER:
                    q.append((curX+dirX, curY+dirY, path+[key]))
        ind+=1
    
    print("DODGE PATH", [enum_2_action[x] for x in dodge_path])
    if dodge_path is None:
        raise "Couldnt dodge"

    og_len = len(dodge_path)
    if len(dodge_path) <= WAIT_TIME:
        dodge_path.append(WAIT)
    
    for i in reversed(range(og_len)):
        if dodge_path[i] == LEFT:
            dodge_path.append(RIGHT)
        elif dodge_path[i] == RIGHT:
            dodge_path.append(LEFT)
        elif dodge_path[i] == UP:
            dodge_path.append(DOWN)
        elif dodge_path[i] == DOWN:
            dodge_path.append(UP)
        
    actions.extend(dodge_path)

    return actions


def bfs_through_crate(ob):
    playerX, playerY = None, None
    for i in range(17):
        for j in range(17):
            if ob[0][i][j] == PLAYER:
                playerX, playerY = i, j
    ogX, ogY = playerX, playerY
    print("player", playerX, playerY)
    q = [(playerX, playerY, [])]
    ind = 0
    visited = set()

    path_to_coin = None
    while ind < len(q):
        curX, curY, path = q[ind]

        if ob[0][curX][curY] == COIN:
            path_to_coin = path
            break

        if (curX, curY) in visited:
            ind+=1
            continue
        else:
            visited.add((curX, curY))

        for (key, (dirX, dirY)) in dirs.items():
            if 0<= curX + dirX < 17 and 0<= curY + dirY < 17:
                val = ob[0][curX+dirX][curY+dirY]
                if  val == FREE or val == COIN or val == CRATE:
                    q.append((curX+dirX, curY+dirY, path+[key]))
        ind+=1
        
    actions = []
    
    for act in path_to_coin:
        playerX += dirs[act][0]
        playerY += dirs[act][1]
        if ob[0][playerX][playerY] == CRATE:
            expect_ob = ob[:]
            expect_ob[0][ogX, ogY] = FREE
            expect_ob[0][playerX - dirs[act][0], playerY - dirs[act][1]] = PLAYER
            actions.extend(place_bomb_and_dodge(expect_ob, playerX - dirs[act][0], playerY - dirs[act][1]))
        else:
            actions.append(act)

    return actions


ob = env.reset()


enum_2_action = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
    4: 'BOMB',
    5: 'WAIT',
}

class ExpertAgent():
    def __init__(self) -> None:
        self.q = []

    def get_action(self, ob):
        if len(self.q) > 0:
            x = self.q[0]
            self.q = self.q[1:]
            return x
        
        acts = bfs_to_coin(ob)
        if len(acts)==0:
            acts = bfs_through_crate(ob)
        
        self.q = acts[1:]
        return acts[0]
  


def count_coins(ob):
    count = 0
    for i in range(17):
        for j in range(17):
            if ob[0][i][j] == COIN:
                count += 1
    
    return count

ob = env.reset()
while True:
    # Get all coins in current zone
    while True:
        acts = bfs_to_coin(ob)
        print("Actions", acts)
        if len(acts)==0:
            break

        for action in acts:
            next_ob, reward, done, info = env.step(action)
            print("Reward", reward, done)
            ob = next_ob
            print("COINS", count_coins(ob))

    
    print("Busting crates")
    acts = bfs_through_crate(ob)

    for action in acts:
        next_ob, reward, done, info = env.step(action)
        print("Reward", reward, done)
        ob = next_ob
        print("COINS", count_coins(ob))

