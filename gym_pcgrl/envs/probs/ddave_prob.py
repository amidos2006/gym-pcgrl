from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, calc_num_regions, get_floor_dist
from gym_pcgrl.envs.probs.ddave.engine import State,BFSAgent,AStarAgent

"""
Generate a fully connected level for a simple platformer similar to Dangerous Dave (http://www.dangerousdave.com)
where the player has to jump at least 2 times to finish
"""
class DDaveProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 11
        self._height = 7
        self._prob = {"empty":0.5, "solid":0.3, "player":0.02, "exit":0.02, "diamond":0.04, "key": 0.02, "spike":0.1}
        self._border_tile = "solid"

        self._solver_power = 5000

        self._max_diamonds = 3
        self._min_spikes = 10

        self._target_jumps = 2
        self._target_solution = 20

        self._rewards = {
            "player": 3,
            "dist-floor": 2,
            "exit": 3,
            "diamonds": 1,
            "key": 3,
            "spikes": 1,
            "regions": 5,
            "num-jumps": 3,
            "dist-win": 0.1,
            "sol-length": 1
        }

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "solid", "player", "exit", "diamond", "key", "spike"]

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid", "player", "exit", "diamond", "key", "spike"
        max_diamonds (int): the maximum amount of diamonds that should be in a level
        min_spikes (int): the minimum amount of spike that should be in a level
        target_jumps (int): the number of jumps needed to consider the game a success
        target_solution (int): the number of moves needed to consider the game a success
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._solver_power = kwargs.get('solver_power', self._solver_power)

        self._max_diamonds = kwargs.get('max_diamonds', self._max_diamonds)
        self._min_spikes = kwargs.get('min_spikes', self._min_spikes)

        self._target_jumps = kwargs.get('target_jumps', self._target_jumps)
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
        dict(string,int): get the status of the best node - "health": player health at that state,
        "airTime": how long before the player start falling, "num_jumps": the number of jumps used till now,
        "col_diamonds": the number of collected diamonds so far, "col_key": the number of collected keys
    """
    def _run_game(self, map):
        gameCharacters=" #@H$V*"
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
        "diamonds": number of diamond tiles, "key": number of key tiles, "spikes": number of spike tiles,
        "reigons": number of connected empty tiles, "num-jumps": number of jumps did by a planning agent,
        "col-diamonds": number of collected diamonds by a planning agent, "dist-win": how close to the win state,
        "sol-length": length of the solution to win the level
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "dist-floor": get_floor_dist(map, ["player"], ["solid"]),
            "exit": calc_certain_tile(map_locations, ["exit"]),
            "diamonds": calc_certain_tile(map_locations, ["diamond"]),
            "key": calc_certain_tile(map_locations, ["key"]),
            "spikes": calc_certain_tile(map_locations, ["spike"]),
            "regions": calc_num_regions(map, map_locations, ["empty","player","diamond","key","exit"]),
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
            "diamonds": get_range_reward(new_stats["diamonds"], old_stats["diamonds"], -np.inf, self._max_diamonds),
            "dist-floor": get_range_reward(new_stats["dist-floor"], old_stats["dist-floor"], 0, 0),
            "key": get_range_reward(new_stats["key"], old_stats["key"], 1, 1),
            "spikes": get_range_reward(new_stats["spikes"], old_stats["spikes"], self._min_spikes, np.inf),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "num-jumps": get_range_reward(new_stats["num-jumps"], old_stats["num-jumps"], np.inf, np.inf),
            "dist-win": get_range_reward(new_stats["dist-win"], old_stats["dist-win"], -np.inf, -np.inf),
            "sol-length": get_range_reward(new_stats["sol-length"], old_stats["sol-length"], np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["dist-floor"] * self._rewards["dist-floor"] +\
            rewards["exit"] * self._rewards["exit"] +\
            rewards["spikes"] * self._rewards["spikes"] +\
            rewards["diamonds"] * self._rewards["diamonds"] +\
            rewards["key"] * self._rewards["key"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["num-jumps"] * self._rewards["num-jumps"] +\
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
                new_stats["num-jumps"] > self._target_jumps

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
            "diamonds": new_stats["diamonds"],
            "key": new_stats["key"],
            "spikes": new_stats["spikes"],
            "regions": new_stats["regions"],
            "col-diamonds": new_stats["col-diamonds"],
            "num-jumps": new_stats["num-jumps"],
            "dist-win": new_stats["dist-win"],
            "sol-length": new_stats["sol-length"]
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using ddave graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/ddave/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/ddave/solid.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/ddave/player.png").convert('RGBA'),
                "exit": Image.open(os.path.dirname(__file__) + "/ddave/exit.png").convert('RGBA'),
                "diamond": Image.open(os.path.dirname(__file__) + "/ddave/diamond.png").convert('RGBA'),
                "key": Image.open(os.path.dirname(__file__) + "/ddave/key.png").convert('RGBA'),
                "spike": Image.open(os.path.dirname(__file__) + "/ddave/spike.png").convert('RGBA')
            }
        return super().render(map)
