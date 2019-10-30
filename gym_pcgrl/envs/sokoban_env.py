from gym_pcgrl.envs.helper import calc_certain_tile, calc_num_regions
from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.probs.sokoban.engine import State,BFSAgent,AStarAgent

class SokobanEnv(PcgrlEnv):
    def _calc_heuristic_solution(self):
        gameCharacters=" #@$."
        int_to_char = dict((i, c) for i, c in enumerate(gameCharacters))
        lvlString = ""
        for x in range(self._rep._map.shape[1]+2):
            lvlString += "#"
        lvlString += "\n"
        for (i,j), index in np.ndenumerate(self._rep._map):
            if j == 0:
                lvlString += "#"
            lvlString += int_to_char[index]
            if j == self._rep._map.shape[1]-1:
                lvlString += "#\n"
        for x in range(self._rep._map.shape[1]+2):
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

    def _calc_rep_stats(self):
        self._rep_stats = {
            "player": calc_certain_tile(self._rep._map, [2]),
            "crate": calc_certain_tile(self._rep._map, [3]),
            "target": calc_certain_tile(self._rep._map, [4]),
            "regions": calc_num_regions(self._rep._map, [0,2,3,4]),
            "dist-win": min(calc_certain_tile(self._rep._map, [3]),calc_certain_tile(self._rep._map, [4]))*(self._rep._width + self._rep._height),
            "sol-length": 0
        }
        if self._rep_stats["player"] == 1:
            if self._rep_stats["crate"] == self._rep_stats["target"] and self._rep_stats["regions"] == 1:
                self._rep_stats["dist-win"], self._rep_stats["sol-length"] = self._calc_heuristic_solution()

    def _init_param(self):
        self._rep._init_param(5, 5, {"0":0.45, "1":0.4, "2": 0.05, "3": 0.05, "4": 0.05})

        self._max_crates = 3

        self._target_solution = 10

        self._rewards = {
            "player": 5,
            "crate": 5,
            "target": 5,
            "regions": 5,
            "ratio": 1,
            "dist-win": 1,
            "sol-length": 1
        }

    def adjust_param(self, **kwargs):
        empty_prob = kwargs.get('empty_prob', self._rep._prob["0"])
        solid_prob = kwargs.get('solid_prob', self._rep._prob["1"])
        player_prob = kwargs.get('player_prob', self._rep._prob["2"])
        crate_prob = kwargs.get('crate_prob', self._rep._prob["3"])
        target_prob = kwargs.get('target_prob', self._rep._prob["4"])
        kwargs["prob"] = {"0":empty_prob, "1":solid_prob, "2":player_prob, "3":crate_prob, "4":target_prob}
        kwargs["width"], kwargs["height"] = kwargs.get('width', self._rep._width), kwargs.get('height', self._rep._height)
        super().adjust_param(**kwargs)

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

    def _calc_total_reward(self, old_stats):
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
        rewards["player"] = old_stats["player"] - self._rep_stats["player"]
        if rewards["player"] > 0 and self._rep_stats["player"] == 0:
            rewards["player"] *= -1
        elif rewards["player"] < 0 and self._rep_stats["player"] == 1:
            rewards["player"] *= -1
        #calculate crate reward (between 1 and max_crates)
        rewards["crate"] = old_stats["crate"] - self._rep_stats["crate"]
        if rewards["crate"] < 0 and old_stats["crate"] == 0:
            rewards["crate"] *= -1
        elif self._rep_stats["crate"] >= 1 and self._rep_stats["crate"] <= self._max_crates and\
                old_stats["crate"] >= 1 and old_stats["crate"] <= self._max_crates:
            rewards["crate"] = 0
        #calculate target reward (between 1 and max_crates)
        rewards["target"] = old_stats["target"] - self._rep_stats["target"]
        if rewards["target"] < 0 and old_stats["target"] == 0:
            rewards["target"] *= -1
        elif self._rep_stats["target"] >= 1 and self._rep_stats["target"] <= self._max_crates and\
                old_stats["target"] >= 1 and old_stats["target"] <= self._max_crates:
            rewards["target"] = 0
        #calculate regions reward
        rewards["regions"] = old_stats["regions"] - self._rep_stats["regions"]
        if self._rep_stats["regions"] == 0 and old_stats["regions"] > 0:
            rewards["regions"] = -1
        #calculate ratio rewards
        new_ratio = abs(self._rep_stats["crate"] - self._rep_stats["target"])
        old_ratio = abs(old_stats["crate"] - old_stats["target"])
        rewards["ratio"] = old_ratio - new_ratio
        #calculate distance remaining to win
        rewards["dist-win"] = old_stats["dist-win"] - self._rep_stats["dist-win"]
        #calculate solution length (more than min solution)
        rewards["sol-length"] = self._rep_stats["sol-length"] - old_stats["sol-length"]
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["crate"] * self._rewards["crate"] +\
            rewards["target"] * self._rewards["target"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["ratio"] * self._rewards["ratio"] +\
            rewards["dist-win"] * self._rewards["dist-win"] +\
            rewards["sol-length"] * self._rewards["sol-length"]

    def _calc_episode_over(self, old_stats):
        return self._rep_stats["sol-length"] >= self._target_solution

    def _calc_debug_info(self, old_stats):
        return {
            "player": self._rep_stats["player"],
            "crate": self._rep_stats["crate"],
            "target": self._rep_stats["target"],
            "regions": self._rep_stats["regions"],
            "dist-win": self._rep_stats["dist-win"],
            "sol-length": self._rep_stats["sol-length"]
        }

    def render(self, mode='human'):
        tile_size = 16
        graphics = {
            "0": Image.open(os.path.dirname(__file__) + "/probs/sokoban/empty.png").convert('RGBA'),
            "1": Image.open(os.path.dirname(__file__) + "/probs/sokoban/solid.png").convert('RGBA'),
            "2": Image.open(os.path.dirname(__file__) + "/probs/sokoban/player.png").convert('RGBA'),
            "3": Image.open(os.path.dirname(__file__) + "/probs/sokoban/crate.png").convert('RGBA'),
            "4": Image.open(os.path.dirname(__file__) + "/probs/sokoban/target.png").convert('RGBA')
        }
        return super().render(graphics, 1, tile_size, mode)
