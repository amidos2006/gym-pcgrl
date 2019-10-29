from gym_pcgrl.envs.helper import calc_certain_tile, calc_num_reachable_tile, calc_num_regions
from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.dungeon.engine import State,BFSAgent,AStarAgent

class DungeonEnv(PcgrlEnv):
    def _calc_heuristic_solution(self):
        gameCharacters=" #@H*$go"
        int_to_char = dict((i, c) for i, c in enumerate(gameCharacters))
        lvlString = ""
        for y in range(self._rep._map.shape[0]+2):
            lvlString += "#"
        lvlString += "\n"
        for (i,j), index in np.ndenumerate(self._rep._map):
            if j == 0:
                lvlString += "#"
            lvlString += int_to_char[index]
            if j == self._rep._map.shape[1]-1:
                lvlString += "#\n"
        for y in range(self._rep._map.shape[0]+2):
            lvlString += "#"
        lvlString += "\n"

        state = State()
        state.stringInitialize(lvlString.split("\n"))

        aStarAgent = AStarAgent()
        bfsAgent = BFSAgent()

        sol,solState,iters = aStarAgent.getSolution(state, 0, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = aStarAgent.getSolution(state, 0.5, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = aStarAgent.getSolution(state, 1, 5000)
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
            "potions": calc_certain_tile(self._rep._map, [4]),
            "treasures": calc_certain_tile(self._rep._map, [5]),
            "enemies": calc_certain_tile(self._rep._map, [6,7]),
            "regions": calc_num_regions(self._rep._map, [0,2,3,4,5,6,7]),
            "reach-exit": 0,
            "col-potions": 0,
            "col-treasures": 0,
            "col-enemies": 0,
            "dist-win": self._rep._width * self._rep._height,
            "sol-length": 0
        }
        if self._rep_stats["player"] == 1:
            self._rep_stats["reach-exit"] = calc_num_reachable_tile(self._rep._map, 2, [0, 2, 3, 4, 5, 6, 7], [3])
            if self._rep_stats["reach-exit"] == 1 and self._rep_stats["regions"] == 1:
                self._rep_stats["dist-win"], self._rep_stats["sol-length"], stats = self._calc_heuristic_solution()
                self._rep_stats["col-potions"] = stats["col_potions"]
                self._rep_stats["col-treasures"] = stats["col_treasures"]
                self._rep_stats["col-enemies"] = stats["col_enemies"]

    def adjust_param(self, **kwargs):
        solid_prob = kwargs.get('solid_prob', 0.4)
        empty_prob = kwargs.get('empty_prob', 0.4)
        player_prob = kwargs.get('player_prob', 0.02)
        exit_prob = kwargs.get('exit_prob', 0.02)
        potion_prob = kwargs.get('potion_prob', 0.03)
        treasure_prob = kwargs.get('treasure_prob', 0.03)
        goblin_prob = kwargs.get('goblin_prob', 0.05)
        ogre_prob = kwargs.get('ogre_prob', 0.05)
        kwargs["prob"] = {"0":empty_prob, "1":solid_prob, "2":player_prob, "3":exit_prob,
                            "4":potion_prob, "5":treasure_prob, "6":goblin_prob, "7": ogre_prob}
        kwargs["width"], kwargs["height"] = kwargs.get('width', 9), kwargs.get('height', 9)
        super().adjust_param(**kwargs)

        self._max_enemies = kwargs.get('max_enemies', 6)
        self._max_potions = kwargs.get('max_potions', 2)
        self._max_treasures = kwargs.get('max_treasures', 3)
        self._min_col_enemies = kwargs.get('min_col_enemies', 0.3)
        self._min_solution = kwargs.get('min_solution', 30)
        self._rewards = {
            "player": kwargs.get("reward_player", 5),
            "exit": kwargs.get("reward_player", 5),
            "potions": kwargs.get("reward_player", 1),
            "treasures": kwargs.get("reward_player", 1),
            "enemies": kwargs.get("reward_enemies", 5),
            "regions": kwargs.get("reward_regions", 5),
            "col-enemies": kwargs.get("reward_col_enemies", 2),
            "dist-win": kwargs.get("reward_dist_win", 1),
            "sol-length": kwargs.get("reward_sol_length", 1)
        }

    def _calc_total_reward(self, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": 0,
            "exit": 0,
            "potions": 0,
            "treasures": 0,
            "enemies": 0,
            "regions": 0,
            "col-enemies": 0,
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
        #calculate enemies reward (between 1 and max_enemies)
        rewards["enemies"] = old_stats["enemies"] - self._rep_stats["enemies"]
        if rewards["enemies"] > 0 and self._rep_stats["enemies"] == 0:
            rewards["enemies"] *= -1
        elif rewards["enemies"] > 0 and self._rep_stats["enemies"] < self._max_enemies:
            rewards["enemies"] = 0
        elif rewards["enemies"] < 0 and old_stats["enemies"] == 0:
            rewards["enemies"] *= -1
        #calculate potions reward (less than max potions)
        rewards["potions"] = old_stats["potions"] - self._rep_stats["potions"]
        if rewards["potions"] > 0 and self._rep_stats["potions"] < self._max_potions:
            rewards["potions"] = 0
        #calculate treasure reward (less than max treasures)
        rewards["treasures"] = old_stats["treasures"] - self._rep_stats["treasures"]
        if rewards["treasures"] > 0 and self._rep_stats["treasures"] < self._max_treasures:
            rewards["treasures"] = 0
        #calculate regions reward (only one region)
        rewards["regions"] = old_stats["regions"] - self._rep_stats["regions"]
        if self._rep_stats["regions"] == 0 and old_stats["regions"] > 0:
            rewards["regions"] = -1
        #calculate ratio of killed enemies
        new_col = self._rep_stats["col-enemies"] / max(self._rep_stats["enemies"], 1)
        old_col = old_stats["col-enemies"] / max(old_stats["enemies"], 1)
        rewards["col-enemies"] = new_col - old_col
        if rewards["col-enemies"] > 0 and old_col >= self._min_col_enemies:
            rewards["col-enemies"] = 0
        if rewards["col-enemies"] < 0 and new_col >= self._min_col_enemies:
            rewards["col-enemies"] = 0
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
            rewards["exit"] * self._rewards["exit"] +\
            rewards["enemies"] * self._rewards["enemies"] +\
            rewards["treasures"] * self._rewards["treasures"] +\
            rewards["potions"] * self._rewards["potions"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["col-enemies"] * self._rewards["col-enemies"] +\
            rewards["dist-win"] * self._rewards["dist-win"] +\
            rewards["sol-length"] * self._rewards["sol-length"]

    def _calc_episode_over(self, old_stats):
        return self._rep_stats["sol-length"] >= self._min_solution and\
                self._rep_stats["enemies"] > 0 and\
                self._rep_stats["col-enemies"] / max(1,self._rep_stats["enemies"]) > self._min_col_enemies

    def _calc_debug_info(self, old_stats):
        return {
            "player": self._rep_stats["player"],
            "exit": self._rep_stats["exit"],
            "potions": self._rep_stats["potions"],
            "treasures": self._rep_stats["treasures"],
            "enemies": self._rep_stats["enemies"],
            "regions": self._rep_stats["regions"],
            "reach-exit": self._rep_stats["reach-exit"],
            "col-potions": self._rep_stats["col-potions"],
            "col-treasures": self._rep_stats["col-treasures"],
            "col-enemies": self._rep_stats["col-enemies"],
            "dist-win": self._rep_stats["dist-win"],
            "sol-length": self._rep_stats["sol-length"]
        }

    def render(self, mode='human'):
        tile_size = 16
        graphics = {
            "0": Image.open(os.path.dirname(__file__) + "/dungeon/empty.png").convert('RGBA'),
            "1": Image.open(os.path.dirname(__file__) + "/dungeon/solid.png").convert('RGBA'),
            "2": Image.open(os.path.dirname(__file__) + "/dungeon/player.png").convert('RGBA'),
            "3": Image.open(os.path.dirname(__file__) + "/dungeon/exit.png").convert('RGBA'),
            "4": Image.open(os.path.dirname(__file__) + "/dungeon/potion.png").convert('RGBA'),
            "5": Image.open(os.path.dirname(__file__) + "/dungeon/treasure.png").convert('RGBA'),
            "6": Image.open(os.path.dirname(__file__) + "/dungeon/goblin.png").convert('RGBA'),
            "7": Image.open(os.path.dirname(__file__) + "/dungeon/ogre.png").convert('RGBA'),
        }
        return super().render(graphics, 1, tile_size, mode)
