from gym_pcgrl.envs.helper import calc_certain_tile, calc_num_reachable_tile, calc_num_regions
from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.sokoban.engine import State,BFSAgent,AStarAgent

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
            "reach-crate": 0,
            "reach-target": 0,
            "dist-win": min(calc_certain_tile(self._rep._map, [3]),calc_certain_tile(self._rep._map, [4]))*(self._rep._width + self._rep._height),
            "sol-length": 0
        }
        if self._rep_stats["player"] == 1:
            self._rep_stats["reach-crate"] = calc_num_reachable_tile(self._rep._map, 2, [0, 2, 3, 4], [3])
            self._rep_stats["reach-target"] = calc_num_reachable_tile(self._rep._map, 2, [0, 2, 3, 4], [4])
            if self._rep_stats["crate"] == self._rep_stats["target"] and\
                self._rep_stats["crate"] == self._rep_stats["reach-crate"] and\
                self._rep_stats["target"] == self._rep_stats["reach-target"]:
                self._rep_stats["dist-win"], self._rep_stats["sol-length"] = self._calc_heuristic_solution()

    def adjust_param(self, **kwargs):
        solid_prob = kwargs.get('solid_prob', 0.4)
        empty_prob = kwargs.get('empty_prob', 0.45)
        player_prob = kwargs.get('player_prob', 0.05)
        crate_prob = kwargs.get('crate_prob', 0.05)
        target_prob = kwargs.get('target_prob', 0.05)
        kwargs["prob"] = {"0":empty_prob, "1":solid_prob, "2":player_prob, "3":crate_prob, "4":target_prob}
        kwargs["width"], kwargs["height"] = kwargs.get('width', 5), kwargs.get('height', 5)
        super().adjust_param(**kwargs)

        self._max_crates = kwargs.get('max_crates', 3)
        self._min_solution = kwargs.get('min_solution', 10)
        self._rewards = {
            "player": kwargs.get('reward_player', 5),
            "crate": kwargs.get('reward_crate', 5),
            "target": kwargs.get('reward_target', 5),
            "regions": kwargs.get('reward_regions', 5),
            "ratio": kwargs.get('reward_ratio', 1),
            "dist-win": kwargs.get('reward_dist_win', 1),
            "sol-length": kwargs.get('reward_sol_length', 1)
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
        #calculate crate reward
        rewards["crate"] = old_stats["crate"] - self._rep_stats["crate"]
        if rewards["crate"] > 0 and self._rep_stats["crate"] == 0:
            rewards["crate"] *= -1
        elif rewards["crate"] > 0 and self._rep_stats["crate"] < self._max_crates:
            rewards["crate"] = 0
        elif rewards["crate"] < 0 and old_stats["crate"] == 0:
            rewards["crate"] *= -1
        #calculate target reward
        rewards["target"] = old_stats["target"] - self._rep_stats["target"]
        if rewards["target"] > 0 and self._rep_stats["target"] == 0:
            rewards["target"] *= -1
        elif rewards["target"] > 0 and self._rep_stats["target"] < self._max_crates:
            rewards["target"] = 0
        elif rewards["target"] < 0 and old_stats["target"] == 0:
            rewards["target"] *= -1
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
        #calculate solution length
        rewards["sol-length"] = self._rep_stats["sol-length"] - old_stats["sol-length"]
        if rewards["sol-length"] > 0 and old_stats["sol-length"] >= self._min_solution:
            rewards["sol-length"] = 0
        elif rewards["sol-length"] < 0 and self._rep_stats["sol-length"] >= self._min_solution:
            rewards["sol-length"] = 0
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["crate"] * self._rewards["crate"] +\
            rewards["target"] * self._rewards["target"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["ratio"] * self._rewards["ratio"] +\
            rewards["dist-win"] * self._rewards["dist-win"] +\
            rewards["sol-length"] * self._rewards["sol-length"]

    def _calc_episode_over(self, old_stats):
        return self._rep_stats["sol-length"] >= self._min_solution

    def _calc_debug_info(self, old_stats):
        return {
            "player": self._rep_stats["player"],
            "crate": self._rep_stats["crate"],
            "target": self._rep_stats["target"],
            "regions": self._rep_stats["regions"],
            "reach-crate": self._rep_stats["reach-crate"],
            "reach-target": self._rep_stats["reach-target"],
            "dist-win": self._rep_stats["dist-win"],
            "sol-length": self._rep_stats["sol-length"]
        }

    def render(self, mode='human'):
        tile_size = 16
        graphics = {
            "0": Image.open(os.path.dirname(__file__) + "/sokoban/empty.png").convert('RGBA'),
            "1": Image.open(os.path.dirname(__file__) + "/sokoban/solid.png").convert('RGBA'),
            "2": Image.open(os.path.dirname(__file__) + "/sokoban/player.png").convert('RGBA'),
            "3": Image.open(os.path.dirname(__file__) + "/sokoban/crate.png").convert('RGBA'),
            "4": Image.open(os.path.dirname(__file__) + "/sokoban/target.png").convert('RGBA')
        }
        return super().render(graphics, 1, tile_size, mode)
