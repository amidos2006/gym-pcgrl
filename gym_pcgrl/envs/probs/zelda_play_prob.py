import numpy as np

from gym_pcgrl.envs.helper import (_get_certain_tiles, calc_certain_tile,
                                   calc_num_regions, get_range_reward,
                                   get_tile_locations)
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem

class Player():
    def __init__(self):
        self.health = 100
        self.keys = 0
        # how many doors have we opened?
        self.doors = 0
        # score or reward
        self.rew = 0
        self.won = 0
        self.coords = (0, 0)
        self.win_time = 0

    def move(self, x_t, y_t):
        self.coords = x_t, y_t

        return True

    def reset(self):
        self.health = 100
        self.keys = 0
        self.doors = 0
        self.rew = 0
        self.won = 0
        self.win_time = 0
        self.coords = (0, 0)

class ZeldaPlayProblem(ZeldaProblem):

    ''' A version of zelda in which a player may control Link and play the game.'''
    def __init__(self, max_step=200):
        super().__init__()
        self._width = self.MAP_X = 8
        self._height = 8
        self.playable = False
        self.active_agent = 0
        # applies only to player turns
        self.player = Player()
        self.win_time = 0
        self.max_step = max_step
        # one key and one door



    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        players = _get_certain_tiles(map_locations, ['player'])
        map_stats = {
            "player": len(players),
            "key": calc_certain_tile(map_locations, ["key"]),
            "door": calc_certain_tile(map_locations, ["door"]),
            "enemies": calc_certain_tile(map_locations, ["bat", "spider", "scorpion"]),
            "regions": calc_num_regions(map, map_locations, ["empty", "player", "key", "bat", "spider", "scorpion"]),
            "nearest-enemy": 0,
            "path-length": 0
        }

        if map_stats["player"] == 1 and map_stats["key"] > 0:#and map_stats["regions"] == 1 and map_stats["key"] >= 1:
            self.playable = True
            self.player.coords = players[0]
            self.polayable = True
       #else:
       #    self.playable = False

        if len(players) > 1:
            self.player.coords = players[-1]

        return map_stats

    def reset(self, rep_stats):
        super().reset(rep_stats)
        self.player.reset()
       #self.playable = False


    def get_reward(self, new_stats, old_stats):
        if self.active_agent == 0:
           #return 0
            return self.get_designer_reward(new_stats, old_stats)

        return self.player.rew

    def move_player(self, trg_chan):
        ''' Moves the player to map coordinates (x_t, y_t).
            Returns True if player can move to target tile.
        '''
        if not self.player.won == 1:
            self.player.win_time += 1

        # impassable tiles

        passable = True
        tile_types = self.get_tile_types()
        player = tile_types.index('player')
        solid = tile_types.index('solid')
        spider = tile_types.index('spider')
        bat = tile_types.index('bat')
        scorpion = tile_types.index('scorpion')
        key = tile_types.index('key')
        door = tile_types.index('door')

        if trg_chan in [solid, player]:
            passable = False

        if trg_chan in [spider, bat, scorpion]:
            self.player.rew -= 0.5
        elif trg_chan == key: # and not self.won:
            self.player.rew += 1
            self.player.keys += 1
           #self._prob.player.rew = self.max_step - self._iteration
           #self.won = True

        if trg_chan == door:
            # door
            if self.player.keys > 0:
                # open door
                self.player.doors += 1
                self.player.keys -= 1
                self.player.rew += 1
                self.player.won = 1
                self.player.rew += self.max_step - self.win_time
            else:
                passable = False
        else:
            self.player.rew = 0

        return passable


    def get_designer_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "key": get_range_reward(new_stats["key"], old_stats["key"], 1, 1),
            "door": get_range_reward(new_stats["door"], old_stats["door"], 1, 1),
            "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], 2, self._max_enemies),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
           #"nearest-enemy": get_range_reward(new_stats["nearest-enemy"], old_stats["nearest-enemy"], self._target_enemy_dist, np.inf),
           #"path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], np.inf, np.inf)
        }
        #calculate the total reward

        return rewards["player"] * self._rewards["player"] +\
            rewards["key"] * self._rewards["key"] +\
            rewards["door"] * self._rewards["door"] +\
            rewards["enemies"] * self._rewards["enemies"] +\
            rewards["regions"] * self._rewards["regions"]#+\
           #rewards["nearest-enemy"] * self._rewards["nearest-enemy"] +\
           #rewards["path-length"] * self._rewards["path-length"]
