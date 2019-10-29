from gym_pcgrl.envs.helper import calc_certain_tile, calc_num_reachable_tile, calc_num_regions
from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.platformer.engine import State,BFSAgent,AStarAgent

class PlatformerEnv(PcgrlEnv):
    def _calc_heuristic_solution(self):
        gameCharacters=" #@H$V*"
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

        sol,solState,iters = aStarAgent.getSolution(state, 1, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = aStarAgent.getSolution(state, 0.5, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = aStarAgent.getSolution(state, 0, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = bfsAgent.getSolution(state, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()

        return solState.getHeuristic(), 0, solState.getGameStatus()

    def _calc_rep_stats(self):
        self._rep_stats = {
            "player": calc_certain_tile(self._rep._map, [2]),
            "exit": calc_certain_tile(self._rep._map, [3]),
            "diamonds": calc_certain_tile(self._rep._map, [4]),
            "key": calc_certain_tile(self._rep._map, [5]),
            "spikes": calc_certain_tile(self._rep._map, [6]),
            "regions": calc_num_regions(self._rep._map, [0,2,3,4,5]),
            "num-jumps": 0,
            "col-diamonds": 0,
            "dist-win": self._rep._width * self._rep._height,
            "sol-length": 0
        }
        if self._rep_stats["player"] == 1:
            if self._rep_stats["exit"] == 1 and self._rep_stats["key"] == 1 and self._rep_stats["regions"] == 1:
                self._rep_stats["dist-win"], self._rep_stats["sol-length"], stats = self._calc_heuristic_solution()
                self._rep_stats["num-jumps"] = stats["num_jumps"]
                self._rep_stats["col-diamonds"] = stats["col_diamonds"]

    def adjust_param(self, **kwargs):
        solid_prob = kwargs.get('solid_prob', 0.3)
        empty_prob = kwargs.get('empty_prob', 0.5)
        player_prob = kwargs.get('player_prob', 0.02)
        exit_prob = kwargs.get('exit_prob', 0.02)
        key_prob = kwargs.get('key_prob', 0.02)
        diamond_prob = kwargs.get('diamond_prob', 0.04)
        spikes_prob = kwargs.get('spikes_prob', 0.1)
        kwargs["prob"] = {"0":empty_prob, "1":solid_prob, "2":player_prob, "3":exit_prob,
                            "4":diamond_prob, "5": key_prob, "6":spikes_prob}
        kwargs["width"], kwargs["height"] = kwargs.get('width', 11), kwargs.get('height', 7)
        super().adjust_param(**kwargs)

        self._max_diamonds = kwargs.get('max_treasures', 3)
        self._min_spikes = kwargs.get('max_spikes', 20)
        self._min_jumps = kwargs.get('min_col_enemies', 2)
        self._min_solution = kwargs.get('min_solution', 20)
        self._rewards = {
            "player": kwargs.get("reward_player", 5),
            "exit": kwargs.get("reward_exit", 5),
            "diamonds": kwargs.get("reward_diamonds", 1),
            "key": kwargs.get("reward_key", 5),
            "spikes": kwargs.get("reward_spikes", 1),
            "regions": kwargs.get("reward_regions", 5),
            "num-jumps": kwargs.get("reward_num_jumps", 2),
            "dist-win": kwargs.get("reward_dist_win", 1),
            "sol-length": kwargs.get("reward_sol_length", 1)
        }

    def _calc_total_reward(self, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": 0,
            "exit": 0,
            "diamonds": 0,
            "key": 0,
            "spikes": 0,
            "regions": 0,
            "num-jumps": 0,
            "dist-win": 0,
            "sol-length": 0
        }
        #calculate the player reward (only one player)
        rewards["player"] = old_stats["player"] - self._rep_stats["player"]
        if rewards["player"] > 0 and self._rep_stats["player"] == 0:
            rewards["player"] *= -1
        elif rewards["player"] < 0 and self._rep_stats["player"] == 1:
            rewards["player"] *= -1
        #calculate the exit reward (only one exit)
        rewards["exit"] = old_stats["exit"] - self._rep_stats["exit"]
        if rewards["exit"] > 0 and self._rep_stats["exit"] == 0:
            rewards["exit"] *= -1
        elif rewards["exit"] < 0 and self._rep_stats["exit"] == 1:
            rewards["exit"] *= -1
        #calculate the key reward (only one key)
        rewards["key"] = old_stats["key"] - self._rep_stats["key"]
        if rewards["key"] > 0 and self._rep_stats["key"] == 0:
            rewards["key"] *= -1
        elif rewards["key"] < 0 and self._rep_stats["key"] == 1:
            rewards["key"] *= -1
        #calculate spike reward (more than min spikes)
        rewards["spikes"] = self._rep_stats["spikes"] - old_stats["spikes"]
        if self._rep_stats["spikes"] > self._min_spikes:
            rewards["spikes"] = 0
        #calculate diamond reward (less than max diamonds)
        rewards["diamonds"] = old_stats["diamonds"] - self._rep_stats["diamonds"]
        if rewards["diamonds"] > 0 and self._rep_stats["diamonds"] < self._max_diamonds:
            rewards["diamonds"] = 0
        #calculate regions reward (only one region)
        rewards["regions"] = old_stats["regions"] - self._rep_stats["regions"]
        if self._rep_stats["regions"] == 0 and old_stats["regions"] > 0:
            rewards["regions"] = -1
        #calculate num jumps reward (more than min jumps)
        rewards["num-jumps"] = self._rep_stats["num-jumps"] - old_stats["num-jumps"]
        if self._rep_stats["num-jumps"] > self._min_jumps:
            rewards["num-jumps"] = 0
        #calculate distance remaining to win
        rewards["dist-win"] = old_stats["dist-win"] - self._rep_stats["dist-win"]
        #calculate solution length
        rewards["sol-length"] = self._rep_stats["sol-length"] - old_stats["sol-length"]
        if self._rep_stats["sol-length"] > self._min_solution:
            rewards["sol-length"] = 0
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["exit"] * self._rewards["exit"] +\
            rewards["spikes"] * self._rewards["spikes"] +\
            rewards["diamonds"] * self._rewards["diamonds"] +\
            rewards["key"] * self._rewards["key"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["num-jumps"] * self._rewards["num-jumps"] +\
            rewards["dist-win"] * self._rewards["dist-win"] +\
            rewards["sol-length"] * self._rewards["sol-length"]

    def _calc_episode_over(self, old_stats):
        return self._rep_stats["sol-length"] >= self._min_solution and\
                self._rep_stats["num-jumps"] > self._min_jumps

    def _calc_debug_info(self, old_stats):
        return {
            "player": self._rep_stats["player"],
            "exit": self._rep_stats["exit"],
            "diamonds": self._rep_stats["diamonds"],
            "key": self._rep_stats["key"],
            "spikes": self._rep_stats["spikes"],
            "regions": self._rep_stats["regions"],
            "col-diamonds": self._rep_stats["col-diamonds"],
            "num-jumps": self._rep_stats["num-jumps"],
            "dist-win": self._rep_stats["dist-win"],
            "sol-length": self._rep_stats["sol-length"]
        }

    def render(self, mode='human'):
        tile_size = 16
        graphics = {
            "0": Image.open(os.path.dirname(__file__) + "/platformer/empty.png").convert('RGBA'),
            "1": Image.open(os.path.dirname(__file__) + "/platformer/solid.png").convert('RGBA'),
            "2": Image.open(os.path.dirname(__file__) + "/platformer/player.png").convert('RGBA'),
            "3": Image.open(os.path.dirname(__file__) + "/platformer/exit.png").convert('RGBA'),
            "4": Image.open(os.path.dirname(__file__) + "/platformer/diamond.png").convert('RGBA'),
            "5": Image.open(os.path.dirname(__file__) + "/platformer/key.png").convert('RGBA'),
            "6": Image.open(os.path.dirname(__file__) + "/platformer/spike.png").convert('RGBA')
        }
        return super().render(graphics, 1, tile_size, mode)
