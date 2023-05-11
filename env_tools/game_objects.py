from settings import game_settings

class Agent(object):
    def __init__(self, id, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.id = id
        self.bombs_left = 3
        self.alive = True
        self.events = []
        self.score = 0

    def update_score(self, points):
        self.score = self.score+points

    def make_bomb(self):
        self.bombs_left -= 1
        return Bomb((self.x, self.y), self, game_settings.bomb_timer+1, game_settings.bomb_power)


class Item(object):
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]


class Bomb(Item):
    def __init__(self, pos, owner, timer, power):
        super(Bomb, self).__init__(pos)
        self.owner = owner
        self.timer = timer
        self.power = power
        self.active = True

    def get_state(self):
        # return ((self.x, self.y), self.timer, self.power, self.active, self.owner.name)
        return (self.x, self.y, self.timer)
    # arena np array, if is -1 hard

    def get_blast_coords(self, arena):
        x, y = self.x, self.y
        blast_coords = [(x, y)]

        for i in range(1, self.power+1):
            if arena[x+i, y] == -1: break
            blast_coords.append((x+i, y))
        for i in range(1, self.power+1):
            if arena[x-i, y] == -1: break
            blast_coords.append((x-i, y))
        for i in range(1, self.power+1):
            if arena[x, y+i] == -1: break
            blast_coords.append((x, y+i))
        for i in range(1, self.power+1):
            if arena[x, y-i] == -1: break
            blast_coords.append((x, y-i))

        return blast_coords


class Coin(Item):
    def __init__(self, pos):
        super(Coin, self).__init__(pos)
        self.collectable = True
        self.collected = False

    def get_state(self):
        return (self.x, self.y)


class Explosion(object):
    def __init__(self, blast_coords, owner, explosion_timer=game_settings.explosion_timer):
        self.blast_coords = blast_coords
        self.owner = owner
        self.timer = explosion_timer
        self.active = True



class Log(object):
    def info(self, message):
        pass
        # print("INFO: "+str(message))

    def debug(self, message):
        pass
        # print("DEBUG: "+str(message))
