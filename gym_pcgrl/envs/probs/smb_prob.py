from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, get_floor_dist, get_type_grouping, get_changes
from gym_pcgrl.envs.probs.smb.engine import State,BFSAgent,AStarAgent


class SMBProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 114
        self._height = 14
        self._prob = {"empty":0.75, "solid":0.1, "enemy":0.01, "brick":0.04, "question":0.01, "coin":0.02, "tube": 0.02}
        self._border_size = (3, 0)

        self._solver_power = 10000

        self._min_empty = 900
        self._min_enemies = 10
        self._max_enemies = 30
        self._min_jumps = 20

        self._rewards = {
            "dist-floor": 2,
            "disjoint-tubes": 1,
            "enemies": 1,
            "empty": 1,
            "noise": 4,
            "jumps": 2,
            "jumps-dist": 2,
            "dist-win": 5
        }

    def get_tile_types(self):
        return ["empty", "solid", "enemy", "brick", "question", "coin", "tube"]

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._min_empty = kwargs.get('min_empty', self._min_empty)
        self._min_enemies = kwargs.get('min_enemies', self._min_enemies)
        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)
        self._min_jumps = kwargs.get('min_jumps', self._min_jumps)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    def _get_runnable_lvl(self, map):
        new_map = []

        for y in range(len(map)):
            new_map.append([])
            for x in range(3):
                if y < self._height - 2:
                    new_map[y].append("empty")
                else:
                    new_map[y].append("solid")
            for x in range(len(map[y])):
                value = map[y][x]
                if map[y][x] == "solid" and y < self._height - 2:
                    value = "solid_above"
                if map[y][x] == "tube":
                    if y >= 1 and map[y-1][x] != "tube":
                        value = "top"
                    if x >= 1 and map[y][x-1] != "tube":
                        value += "_left"
                    else:
                        value += "_right"
                new_map[y].append(value)
            for x in range(3):
                if y < self._height - 2:
                    new_map[y].append("empty")
                else:
                    new_map[y].append("solid")

        new_map[-3][1] = "player"
        new_map[-3][-2] = "solid_above"
        for y in range(3, len(map) - 3):
            new_map[y][-2] = "pole"
        new_map[1][-2] = "pole_top"
        new_map[2][-2] = "pole_flag"
        new_map[2][-3] = "flag"

        return new_map

    def _run_game(self, map):
        gameCharacters=" # ## #"
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvlString = ""
        for i in range(len(map)):
            if i < self._height - 3:
                lvlString += "   "
            elif i == self._height - 3:
                lvlString += " @ "
            else:
                lvlString += "###"
            for j in range(len(map[i])):
                string = map[i][j]
                lvlString += string_to_char[string]
            if i < self._height - 3:
                lvlString += " | "
            elif i == self._height - 3:
                lvlString += " # "
            else:
                lvlString += "###"
            lvlString += "\n"

        state = State()
        state.stringInitialize(lvlString.split("\n"))

        aStarAgent = AStarAgent()

        sol,solState,iters = aStarAgent.getSolution(state, 1, self._solver_power)
        if solState.checkWin():
            return 0, solState.getGameStatus()
        sol,solState,iters = aStarAgent.getSolution(state, 0, self._solver_power)
        if solState.checkWin():
            return 0, solState.getGameStatus()

        return solState.getHeuristic(), solState.getGameStatus()

    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "dist-floor": get_floor_dist(map, ["enemy"], ["solid", "brick", "question", "tube_left", "tube_right"]),
            "disjoint-tubes": get_type_grouping(map, ["tube"], [(-1,0),(1,0)],1,1),
            "enemies": calc_certain_tile(map_locations, ["enemy"]),
            "empty": calc_certain_tile(map_locations, ["empty"]),
            "noise": get_changes(map, False) + get_changes(map, True),
            "jumps": 0,
            "jumps-dist": 0,
            "dist-win": 0
        }
        map_stats["dist-win"], play_stats = self._run_game(map)
        map_stats["jumps"] = play_stats["jumps"]
        prev_jump = 0
        value = 0
        for l in play_stats["jump_locs"]:
            value = max(value, l[0] - prev_jump)
            prev_jump = l[0]
        value = max(value, self._width - prev_jump)
        map_stats["jumps-dist"] = value
        return map_stats

    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "dist-floor": get_range_reward(new_stats["dist-floor"], old_stats["dist-floor"], 0, 0),
            "disjoint-tubes": get_range_reward(new_stats["disjoint-tubes"], old_stats["disjoint-tubes"], 0, 0),
            "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], self._min_enemies, self._max_enemies),
            "empty": get_range_reward(new_stats["empty"], old_stats["empty"], self._min_empty, np.inf),
            "noise": get_range_reward(new_stats["noise"], old_stats["noise"], 0, 0),
            "jumps": get_range_reward(new_stats["jumps"], old_stats["jumps"], self._min_jumps, np.inf),
            "jumps-dist": get_range_reward(new_stats["jumps-dist"], old_stats["jumps-dist"], 0, 0),
            "dist-win": get_range_reward(new_stats["dist-win"], old_stats["dist-win"], 0, 0)
        }
        #calculate the total reward
        return rewards["dist-floor"] * self._rewards["dist-floor"] +\
            rewards["disjoint-tubes"] * self._rewards["disjoint-tubes"] +\
            rewards["enemies"] * self._rewards["enemies"] +\
            rewards["empty"] * self._rewards["empty"] +\
            rewards["noise"] * self._rewards["noise"] +\
            rewards["jumps"] * self._rewards["jumps"] +\
            rewards["jumps-dist"] * self._rewards["jumps-dist"] +\
            rewards["dist-win"] * self._rewards["dist-win"]

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["dist-win"] <= 0

    def get_debug_info(self, new_stats, old_stats):
        return {
            "dist-floor": new_stats["dist-floor"],
            "disjoint-tubes": new_stats["disjoint-tubes"],
            "enemies": new_stats["enemies"],
            "empty": new_stats["empty"],
            "noise": new_stats["noise"],
            "jumps": new_stats["jumps"],
            "jumps-dist": new_stats["jumps-dist"],
            "dist-win": new_stats["dist-win"]
        }

    def render(self, map):
        new_map = self._get_runnable_lvl(map)

        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/smb/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/smb/solid_floor.png").convert('RGBA'),
                "solid_above": Image.open(os.path.dirname(__file__) + "/smb/solid_air.png").convert('RGBA'),
                "enemy": Image.open(os.path.dirname(__file__) + "/smb/enemy.png").convert('RGBA'),
                "brick": Image.open(os.path.dirname(__file__) + "/smb/brick.png").convert('RGBA'),
                "question": Image.open(os.path.dirname(__file__) + "/smb/question.png").convert('RGBA'),
                "coin": Image.open(os.path.dirname(__file__) + "/smb/coin.png").convert('RGBA'),
                "top_left": Image.open(os.path.dirname(__file__) + "/smb/top_left.png").convert('RGBA'),
                "top_right": Image.open(os.path.dirname(__file__) + "/smb/top_right.png").convert('RGBA'),
                "tube_left": Image.open(os.path.dirname(__file__) + "/smb/tube_left.png").convert('RGBA'),
                "tube_right": Image.open(os.path.dirname(__file__) + "/smb/tube_right.png").convert('RGBA'),
                "pole_top": Image.open(os.path.dirname(__file__) + "/smb/poletop.png").convert('RGBA'),
                "pole": Image.open(os.path.dirname(__file__) + "/smb/pole.png").convert('RGBA'),
                "pole_flag": Image.open(os.path.dirname(__file__) + "/smb/flag.png").convert('RGBA'),
                "flag": Image.open(os.path.dirname(__file__) + "/smb/flagside.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/smb/player.png").convert('RGBA')
            }
        self._border_size = (0, 0)
        img = super().render(new_map)
        self._border_size = (3, 0)
        return img
