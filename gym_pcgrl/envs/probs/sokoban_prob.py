import os
from PIL import Image
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.probs.helper import calc_certain_tile, calc_num_regions
from gym_pcgrl.envs.probs.sokoban.engine import State,BFSAgent,AStarAgent

class SokobanProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 5
        self._height = 5
        self._prob = {"empty":0.45, "solid":0.4, "player": 0.05, "crate": 0.05, "target": 0.05}

        self._max_crates = 3

        self._target_solution = 20

        self._rewards = {
            "player": 5,
            "crate": 5,
            "target": 5,
            "regions": 5,
            "ratio": 1,
            "dist-win": 1,
            "sol-length": 1
        }

    def get_tile_types(self):
        return ["empty", "solid", "player", "crate", "target"]

    def adjust_param(self, **kwargs):
        self._width, self._height = kwargs.get('width', self._width), kwargs.get('height', self._height)
        self._prob["empty"] = kwargs.get('empty_prob', self._prob["empty"])
        self._prob["solid"] = kwargs.get('solid_prob', self._prob["solid"])
        self._prob["player"] = kwargs.get('player_prob', self._prob["player"])
        self._prob["crate"] = kwargs.get('crate_prob', self._prob["crate"])
        self._prob["target"] = kwargs.get('target_prob', self._prob["target"])

        self._max_crates = kwargs.get('max_crates', self._max_crates)

        self._target_solution = kwargs.get('min_solution', self._target_solution)

        self._rewards = {
            "player": kwargs.get('reward_player', self._rewards["player"]),
            "crate": kwargs.get('reward_crate', self._rewards["crate"]),
            "target": kwargs.get('reward_target', self._rewards["target"]),
            "regions": kwargs.get('reward_regions', self._rewards["regions"]),
            "ratio": kwargs.get('reward_ratio', self._rewards["ratio"]),
            "dist-win": kwargs.get('reward_dist_win', self._rewards["dist-win"]),
            "sol-length": kwargs.get('reward_sol_length', self._rewards["sol-length"])
        }

    def _run_game(self, map):
        gameCharacters=" #@$."
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvlString = ""
        for x in range(self._width+2):
            lvlString += "#"
        lvlString += "\n"
        for i in range(len(map)):
            for j in range(len(map[i])):
                string = map[i][j]
                if j == 0:
                    lvlString += "#"
                lvlString += string_to_char[string]
                if j == self._width-1:
                    lvlString += "#\n"
        for x in range(self._width+2):
            lvlString += "#"
        lvlString += "\n"

        state = State()
        state.stringInitialize(lvlString.split("\n"))

        aStarAgent = AStarAgent()
        bfsAgent = BFSAgent()

        sol,solState,iters = bfsAgent.getSolution(state, 5000)
        if solState.checkWin():
            return 0, len(sol)
        sol,solState,iters = aStarAgent.getSolution(state, 1, 5000)
        if solState.checkWin():
            return 0, len(sol)
        sol,solState,iters = aStarAgent.getSolution(state, 0.5, 5000)
        if solState.checkWin():
            return 0, len(sol)
        sol,solState,iters = aStarAgent.getSolution(state, 0.25, 5000)
        if solState.checkWin():
            return 0, len(sol)
        sol,solState,iters = aStarAgent.getSolution(state, 0, 5000)
        if solState.checkWin():
            return 0, len(sol)
        return solState.getHeuristic(), 0

    def get_stats(self, map):
        map_stats = {
            "player": calc_certain_tile(map, ["player"]),
            "crate": calc_certain_tile(map, ["crate"]),
            "target": calc_certain_tile(map, ["target"]),
            "regions": calc_num_regions(map, ["empty","player","crate","target"]),
            "dist-win": self._width * self._height * (self._width + self._height),
            "sol-length": 0
        }
        if map_stats["player"] == 1:
            if map_stats["crate"] == map_stats["target"] and map_stats["regions"] == 1:
                map_stats["dist-win"], map_stats["sol-length"] = self._run_game(map)
        return map_stats

    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": 0,
            "crate": 0,
            "target": 0,
            "regions": 0,
            "ratio": 0,
            "dist-win": 0,
            "sol-length": 0
        }
        #calculate the player reward
        rewards["player"] = old_stats["player"] - new_stats["player"]
        if rewards["player"] > 0 and new_stats["player"] == 0:
            rewards["player"] *= -1
        elif rewards["player"] < 0 and new_stats["player"] == 1:
            rewards["player"] *= -1
        #calculate crate reward (between 1 and max_crates)
        rewards["crate"] = old_stats["crate"] - new_stats["crate"]
        if rewards["crate"] < 0 and old_stats["crate"] == 0:
            rewards["crate"] *= -1
        elif new_stats["crate"] >= 1 and new_stats["crate"] <= self._max_crates and\
                old_stats["crate"] >= 1 and old_stats["crate"] <= self._max_crates:
            rewards["crate"] = 0
        #calculate target reward (between 1 and max_crates)
        rewards["target"] = old_stats["target"] - new_stats["target"]
        if rewards["target"] < 0 and old_stats["target"] == 0:
            rewards["target"] *= -1
        elif new_stats["target"] >= 1 and new_stats["target"] <= self._max_crates and\
                old_stats["target"] >= 1 and old_stats["target"] <= self._max_crates:
            rewards["target"] = 0
        #calculate regions reward
        rewards["regions"] = old_stats["regions"] - new_stats["regions"]
        if new_stats["regions"] == 0 and old_stats["regions"] > 0:
            rewards["regions"] = -1
        #calculate ratio rewards
        new_ratio = abs(new_stats["crate"] - new_stats["target"])
        old_ratio = abs(old_stats["crate"] - old_stats["target"])
        rewards["ratio"] = old_ratio - new_ratio
        #calculate distance remaining to win
        rewards["dist-win"] = old_stats["dist-win"] - new_stats["dist-win"]
        #calculate solution length (more than min solution)
        rewards["sol-length"] = new_stats["sol-length"] - old_stats["sol-length"]
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["crate"] * self._rewards["crate"] +\
            rewards["target"] * self._rewards["target"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["ratio"] * self._rewards["ratio"] +\
            rewards["dist-win"] * self._rewards["dist-win"] +\
            rewards["sol-length"] * self._rewards["sol-length"]

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["sol-length"] >= self._target_solution

    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "crate": new_stats["crate"],
            "target": new_stats["target"],
            "regions": new_stats["regions"],
            "dist-win": new_stats["dist-win"],
            "sol-length": new_stats["sol-length"]
        }

    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/sokoban/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/sokoban/solid.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/sokoban/player.png").convert('RGBA'),
                "crate": Image.open(os.path.dirname(__file__) + "/sokoban/crate.png").convert('RGBA'),
                "target": Image.open(os.path.dirname(__file__) + "/sokoban/target.png").convert('RGBA')
            }
        return super().render(map)
