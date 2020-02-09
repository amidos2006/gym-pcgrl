from gym_pcgrl.envs.helper import (_get_certain_tiles, calc_certain_tile,
                                   get_tile_locations, calc_num_regions, get_range_reward)
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem


class ZeldaPlayProblem(ZeldaProblem):
    ''' A version of zelda in which a player may control Link and play the game.'''
    def __init__(self):
        super().__init__()
        self._width = 8
        self._height = 8
        self.playable = False
        self.active_agent = 0
        # applies only to player turns
        self.player_coords = None
        self.play_rew = 0

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

        if map_stats["player"] > 0 and map_stats["key"] > 0:#and map_stats["regions"] == 1 and map_stats["key"] >= 1:
            self.playable = True
       #else:
       #    self.playable = False
        if len(players) > 0:
            self.player_coords = players[0]

        return map_stats

    def reset(self, rep_stats):
        super().reset(rep_stats)
        self.play_rew = 0
       #self.playable = False


    def get_reward(self, new_stats, old_stats):
        if self.active_agent == 0:
           #return 0
            return self.get_designer_reward(new_stats, old_stats)
        return self.play_rew



    def get_designer_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "key": get_range_reward(new_stats["key"], old_stats["key"], 1, 64),
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
