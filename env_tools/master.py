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


def bfs_to_coin(ob):
    playerX, playerY = None, None
    for i in range(17):
        for j in range(17):
            if ob[0][i][j] == PLAYER:
                playerX, playerY = i, j

    print(playerX, playerY)
    q = [(playerX,playerY, [])]
    ind = 0
    visited = set()
    dirs = {
        LEFT: (0, -1),
        RIGHT: (0, 1),
        UP: (-1, 0),
        DOWN: (1, 0)
    }

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
                if  val != WALL or val != CRATE:
                    q.append((curX+dirX, curY+dirY, path+[key]))
    
        ind+=1

    print("No coin found")



c=10
ob = env.reset()
while c:=c-1>0:
    acts = bfs_to_coin(ob)
    print(acts)
    if len(acts)==0:
        break

    for action in acts:
        next_ob, reward, done, info = env.step(action)
        print(reward)
        ob = next_ob