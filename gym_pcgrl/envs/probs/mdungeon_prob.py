import os
import numpy as np
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, calc_num_regions
from gym_pcgrl.envs.probs.mdungeon.engine import State,BFSAgent,AStarAgent

"""
Generate a fully connected level for a simple dungeon crawler similar to MiniDungeons 1 (http://minidungeons.com/)
where the player has to kill 50% of enemies before done
"""
class MDungeonProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 7
        self._height = 11
        self._prob = {"empty":0.4, "solid": 0.4, "player":0.02, "exit":0.02, "potion":0.03, "treasure":0.03, "goblin":0.05, "ogre": 0.05}
        self._border_tile = "solid"

        self._solver_power = 5000

        self._max_enemies = 6
        self._max_potions = 2
        self._max_treasures = 3

        self._target_col_enemies = 0.5
        self._target_solution = 20

        self._rewards = {
            "player": 3,
            "exit": 3,
            "potions": 1,
            "treasures": 1,
            "enemies": 2,
            "regions": 5,
            "col-enemies": 2,
            "dist-win": 0.1,
            "sol-length": 1
        }

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "solid", "player", "exit", "potion", "treasure", "goblin", "ogre"]

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid", "player", "exit","potion",
        "treasure", "goblin", "ogre"
        max_enemies (int): the max amount of enemies that should appear in a level
        max_potions (int): the max amount of potions that should appear in a level
        max_treasures (int): the max amount of treasure that should appear in a level
        target_col_enemies (int): the target amount of killed enemies that the game is considered a success
        target_solution (int): the minimum amount of movement needed to consider the level a success
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._solver_power = kwargs.get('solver_power', self._solver_power)

        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)
        self._max_potions = kwargs.get('max_potions', self._max_potions)
        self._max_treasures = kwargs.get('max_treasures', self._max_treasures)

        self._target_col_enemies = kwargs.get('target_col_enemies', self._target_col_enemies)
        self._target_solution = kwargs.get('target_solution', self._target_solution)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    """
    Private function that runs the game on the input level

    Parameters:
        map (string[][]): the input level to run the game on

    Returns:
        float: how close you are to winning (0 if you win)
        int: the solution length if you win (0 otherwise)
        dict(string,int): get the status of the best node - "health": the current player health,
        "col_treasures": number of collected treasures, "col_potions": number of collected potions,
        "col_enemies": number of killed enemies
    """
    def _run_game(self, map):
        gameCharacters=" #@H*$go"
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

        sol,solState,iters = aStarAgent.getSolution(state, 1, self._solver_power)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = aStarAgent.getSolution(state, 0.5, self._solver_power)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = aStarAgent.getSolution(state, 0, self._solver_power)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = bfsAgent.getSolution(state, self._solver_power)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()

        return solState.getHeuristic(), 0, solState.getGameStatus()

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "player": number of player tiles, "exit": number of exit tiles,
        "potions": number of potion tiles, "treasures": number of treasure tiles, "enemies": number of goblin and ogre tiles,
        "reigons": number of connected empty tiles, "col-potions": number of collected potions by a planning agent,
        "col-treasures": number of collected treasures by a planning agent, "col-enemies": number of killed enemies by a planning agent,
        "dist-win": how close to the win state, "sol-length": length of the solution to win the level
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "exit": calc_certain_tile(map_locations, ["exit"]),
            "potions": calc_certain_tile(map_locations, ["potion"]),
            "treasures": calc_certain_tile(map_locations, ["treasure"]),
            "enemies": calc_certain_tile(map_locations, ["goblin","ogre"]),
            "regions": calc_num_regions(map, map_locations, ["empty","player","exit","potion","treasure","goblin","ogre"]),
            "col-potions": 0,
            "col-treasures": 0,
            "col-enemies": 0,
            "dist-win": self._width * self._height,
            "sol-length": 0
        }
        if map_stats["player"] == 1 and map_stats["exit"] == 1 and map_stats["regions"] == 1:
                map_stats["dist-win"], map_stats["sol-length"], play_stats = self._run_game(map)
                map_stats["col-potions"] = play_stats["col_potions"]
                map_stats["col-treasures"] = play_stats["col_treasures"]
                map_stats["col-enemies"] = play_stats["col_enemies"]
        return map_stats

    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "exit": get_range_reward(new_stats["exit"], old_stats["exit"], 1, 1),
            "potions": get_range_reward(new_stats["potions"], old_stats["potions"], -np.inf, self._max_potions),
            "treasures": get_range_reward(new_stats["treasures"], old_stats["treasures"], -np.inf, self._max_treasures),
            "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], 1, self._max_enemies),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "col-enemies": get_range_reward(new_stats["col-enemies"], old_stats["col-enemies"], np.inf, np.inf),
            "dist-win": get_range_reward(new_stats["dist-win"], old_stats["dist-win"], -np.inf, -np.inf),
            "sol-length": get_range_reward(new_stats["sol-length"], old_stats["sol-length"], np.inf, np.inf)
        }
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

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, old_stats):
        return new_stats["sol-length"] >= self._target_solution and\
                new_stats["enemies"] > 0 and\
                new_stats["col-enemies"] / max(1,new_stats["enemies"]) > self._target_col_enemies

    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "exit": new_stats["exit"],
            "potions": new_stats["potions"],
            "treasures": new_stats["treasures"],
            "enemies": new_stats["enemies"],
            "regions": new_stats["regions"],
            "col-potions": new_stats["col-potions"],
            "col-treasures": new_stats["col-treasures"],
            "col-enemies": new_stats["col-enemies"],
            "dist-win": new_stats["dist-win"],
            "sol-length": new_stats["sol-length"]
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using mdungeon graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/mdungeon/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/mdungeon/solid.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/mdungeon/player.png").convert('RGBA'),
                "exit": Image.open(os.path.dirname(__file__) + "/mdungeon/exit.png").convert('RGBA'),
                "potion": Image.open(os.path.dirname(__file__) + "/mdungeon/potion.png").convert('RGBA'),
                "treasure": Image.open(os.path.dirname(__file__) + "/mdungeon/treasure.png").convert('RGBA'),
                "goblin": Image.open(os.path.dirname(__file__) + "/mdungeon/goblin.png").convert('RGBA'),
                "ogre": Image.open(os.path.dirname(__file__) + "/mdungeon/ogre.png").convert('RGBA'),
            }
        return super().render(map)
