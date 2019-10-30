from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.probs.helper import calc_certain_tile, calc_num_regions
from gym_pcgrl.envs.probs.ddave.engine import State,BFSAgent,AStarAgent

class DDaveProblem(Problem):
    def __init__(self):
        super().__init__()

        self._width = 11
        self._height = 7
        self._prob = {"0":0.5, "1":0.3, "2":0.02, "3":0.02, "4":0.04, "5": 0.02, "6":0.1}

        self._border_size = 1
        self._border_tile = 1
        self._tile_size = 16
        self._graphics = None

        self._max_diamonds = 3
        self._min_spikes = 20

        self._target_jumps = 2
        self._target_solution = 20

        self._rewards = {
            "player": 5,
            "exit": 5,
            "diamonds": 1,
            "key": 5,
            "spikes": 1,
            "regions": 5,
            "num-jumps": 2,
            "dist-win": 1,
            "sol-length": 1
        }

    def adjust_param(self, **kwargs):
        self._width, self._height = kwargs.get('width', self._width), kwargs.get('height', self._height)
        self._prob["0"] = kwargs.get('empty_prob', self._prob["0"])
        self._prob["1"] = kwargs.get('solid_prob', self._prob["1"])
        self._prob["2"] = kwargs.get('player_prob', self._prob["2"])
        self._prob["3"] = kwargs.get('exit_prob', self._prob["3"])
        self._prob["4"] = kwargs.get('diamond_prob', self._prob["4"])
        self._prob["5"] = kwargs.get('key_prob', self._prob["5"])
        self._prob["6"] = kwargs.get('spikes_prob', self._prob["6"])
        kwargs["prob"] = self._prob

        self._max_diamonds = kwargs.get('max_diamonds', self._max_diamonds)
        self._min_spikes = kwargs.get('min_spikes', self._min_spikes)

        self._target_jumps = kwargs.get('target_jumps', self._target_jumps)
        self._target_solution = kwargs.get('target_solution', self._target_solution)

        self._rewards = {
            "player": kwargs.get("reward_player", self._rewards["player"]),
            "exit": kwargs.get("reward_exit", self._rewards["exit"]),
            "diamonds": kwargs.get("reward_diamonds", self._rewards["diamonds"]),
            "key": kwargs.get("reward_key", self._rewards["key"]),
            "spikes": kwargs.get("reward_spikes", self._rewards["spikes"]),
            "regions": kwargs.get("reward_regions", self._rewards["regions"]),
            "num-jumps": kwargs.get("reward_num_jumps", self._rewards["num-jumps"]),
            "dist-win": kwargs.get("reward_dist_win", self._rewards["dist-win"]),
            "sol-length": kwargs.get("reward_sol_length", self._rewards["sol-length"])
        }

    def _run_game(self, map):
        gameCharacters=" #@H$V*"
        int_to_char = dict((i, c) for i, c in enumerate(gameCharacters))
        lvlString = ""
        for x in range(self._width+2):
            lvlString += "#"
        lvlString += "\n"
        for (i,j), index in np.ndenumerate(map):
            if j == 0:
                lvlString += "#"
            lvlString += int_to_char[index]
            if j == self._width-1:
                lvlString += "#\n"
        for x in range(self._width+2):
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

    def get_stats(self, map):
        map_stats = {
            "player": calc_certain_tile(map, [2]),
            "exit": calc_certain_tile(map, [3]),
            "diamonds": calc_certain_tile(map, [4]),
            "key": calc_certain_tile(map, [5]),
            "spikes": calc_certain_tile(map, [6]),
            "regions": calc_num_regions(map, [0,2,3,4,5]),
            "num-jumps": 0,
            "col-diamonds": 0,
            "dist-win": self._width * self._height,
            "sol-length": 0
        }
        if map_stats["player"] == 1:
            if map_stats["exit"] == 1 and map_stats["key"] == 1 and map_stats["regions"] == 1:
                map_stats["dist-win"], map_stats["sol-length"], play_stats = self._run_game(map)
                map_stats["num-jumps"] = play_stats["num_jumps"]
                map_stats["col-diamonds"] = play_stats["col_diamonds"]
        return map_stats

    def get_reward(self, new_stats, old_stats):
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
        rewards["player"] = old_stats["player"] - new_stats["player"]
        if rewards["player"] > 0 and new_stats["player"] == 0:
            rewards["player"] *= -1
        elif rewards["player"] < 0 and new_stats["player"] == 1:
            rewards["player"] *= -1
        #calculate the exit reward (only one exit)
        rewards["exit"] = old_stats["exit"] - new_stats["exit"]
        if rewards["exit"] > 0 and new_stats["exit"] == 0:
            rewards["exit"] *= -1
        elif rewards["exit"] < 0 and new_stats["exit"] == 1:
            rewards["exit"] *= -1
        #calculate the key reward (only one key)
        rewards["key"] = old_stats["key"] - new_stats["key"]
        if rewards["key"] > 0 and new_stats["key"] == 0:
            rewards["key"] *= -1
        elif rewards["key"] < 0 and new_stats["key"] == 1:
            rewards["key"] *= -1
        #calculate spike reward (more than min spikes)
        rewards["spikes"] = new_stats["spikes"] - old_stats["spikes"]
        if new_stats["spikes"] >= self._min_spikes and old_stats["spikes"] >= self._min_spikes:
            rewards["spikes"] = 0
        #calculate diamond reward (less than max diamonds)
        rewards["diamonds"] = old_stats["diamonds"] - new_stats["diamonds"]
        if new_stats["diamonds"] <= self._max_diamonds and old_stats["diamonds"] <= self._max_diamonds:
            rewards["diamonds"] = 0
        #calculate regions reward (only one region)
        rewards["regions"] = old_stats["regions"] - new_stats["regions"]
        if new_stats["regions"] == 0 and old_stats["regions"] > 0:
            rewards["regions"] = -1
        #calculate num jumps reward (more than min jumps)
        rewards["num-jumps"] = new_stats["num-jumps"] - old_stats["num-jumps"]
        #calculate distance remaining to win
        rewards["dist-win"] = old_stats["dist-win"] - new_stats["dist-win"]
        #calculate solution length
        rewards["sol-length"] = new_stats["sol-length"] - old_stats["sol-length"]
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

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["sol-length"] >= self._target_solution and\
                new_stats["num-jumps"] > self._target_jumps

    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "exit": new_stats["exit"],
            "diamonds": new_stats["diamonds"],
            "key": new_stats["key"],
            "spikes": new_stats["spikes"],
            "regions": new_stats["regions"],
            "col-diamonds": new_stats["col-diamonds"],
            "num-jumps": new_stats["num-jumps"],
            "dist-win": new_stats["dist-win"],
            "sol-length": new_stats["sol-length"]
        }

    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "0": Image.open(os.path.dirname(__file__) + "/ddave/empty.png").convert('RGBA'),
                "1": Image.open(os.path.dirname(__file__) + "/ddave/solid.png").convert('RGBA'),
                "2": Image.open(os.path.dirname(__file__) + "/ddave/player.png").convert('RGBA'),
                "3": Image.open(os.path.dirname(__file__) + "/ddave/exit.png").convert('RGBA'),
                "4": Image.open(os.path.dirname(__file__) + "/ddave/diamond.png").convert('RGBA'),
                "5": Image.open(os.path.dirname(__file__) + "/ddave/key.png").convert('RGBA'),
                "6": Image.open(os.path.dirname(__file__) + "/ddave/spike.png").convert('RGBA')
            }
        return super().render(map)
