from gym_pcgrl.envs.helper import (_get_certain_tiles, calc_certain_tile,
                                   get_tile_locations, calc_num_regions)
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem


class ZeldaPlayProblem(ZeldaProblem):
    ''' A version of zelda in which a player may control Link and play the game.'''
    def __init__(self):
        super().__init__()
        self._width = 16
        self._height = 16
        self.playable = False
        self.active_agent = 0
        # applies only to player turns
        self.player_coords = (0, 0)

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

        if map_stats["player"] == 1 and map_stats["regions"] == 1 and map_stats["key"] >= 1:
            self.playable = True
            self.player_coords = players[0]

        return map_stats


    def get_reward(self, new_stats, old_stats):
        if self.active_agent == 0:
            return super().get_reward(new_stats, old_stats)
        return self.player_reward

       #if map_stats["player"] == 1 and map_stats["regions"] == 1:
       #    p_x,p_y = map_locations["player"][0]
       #    enemies = []
       #    enemies.extend(map_locations["spider"])
       #    enemies.extend(map_locations["bat"])
       #    enemies.extend(map_locations["scorpion"])
       #    if len(enemies) > 0:
       #        dikjstra,_ = run_dikjstra(p_x, p_y, map, ["empty", "player", "bat", "spider", "scorpion"])
       #        min_dist = self._width * self._height
       #        for e_x,e_y in enemies:
       #            if dikjstra[e_y][e_x] > 0 and dikjstra[e_y][e_x] < min_dist:
       #                min_dist = dikjstra[e_y][e_x]
       #        map_stats["nearest-enemy"] = min_dist
       #    if map_stats["key"] == 1 and map_stats["door"] == 1:
       #        k_x,k_y = map_locations["key"][0]
       #        d_x,d_y = map_locations["door"][0]
       #        dikjstra,_ = run_dikjstra(p_x, p_y, map, ["empty", "key", "player", "bat", "spider", "scorpion"])
       #        map_stats["path-length"] += dikjstra[k_y][k_x]
       #        dikjstra,_ = run_dikjstra(k_x, k_y, map, ["empty", "player", "key", "door", "bat", "spider", "scorpion"])
       #        map_stats["path-length"] += dikjstra[d_y][d_x]
