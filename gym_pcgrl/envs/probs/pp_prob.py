import os
import numpy as np
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_longest_path

from gym_pcgrl.envs.probs.astar import aStar, Node
import tensorflow as tf

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class PathPlanningProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 8
        self._height = 8
        self._prob = {"empty": 0.5, "solid":0.5}
        self._border_tile = "solid"

        self._start = (self._height - 1,0)
        # todo : x, y 좌표 고려
        self._end = (0, self._width - 1)

        self._target_length = 25
        self._random_probs = True

        self._rewards = {
            "regions": 3,
            "arrived": 5,
            "pp_length" : 1,
            "not_arrived_length": -0.1
        }

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "solid"]

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid"
        target_path (int): the current path length that the episode turn when it reaches
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        # todo : target path를 도착하는데까지 걸리는 path 길이로 변경
        self._target_length = kwargs.get('target_path', self._target_length)
        self._random_probs = kwargs.get('random_probs', self._random_probs)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    """
    Resets the problem to the initial state and save the start_stats from the starting map.
    Also, it can be used to change values between different environment resets

    Parameters:
        start_stats (dict(string,any)): the first stats of the map
    """
    def reset(self, start_stats):
        super().reset(start_stats)
        if self._random_probs:
            self._prob["empty"] = self._random.random()
            self._prob["solid"] = 1 - self._prob["empty"]

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map):
        binary_map = np.zeros_like(map)
        for i in range(binary_map.shape[0]):
            for j in range(binary_map.shape[1]):
                if map[i][j] == 'solid':
                    binary_map[i][j] = 1
                else:
                    binary_map[i][j] = 0

        map_locations = get_tile_locations(map, self.get_tile_types())
        arrived, pp_length, not_arrived_length = aStar(binary_map, self._start, self._end)

        return {
            "regions": calc_num_regions(map, map_locations, ["empty"]),
            "arrived" : arrived,
            "pp_length" : pp_length,
            "not_arrived_length" : not_arrived_length
        }

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
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "arrived" : get_range_reward(new_stats["arrived"], old_stats["arrived"], 1, 1),
            "pp_length": get_range_reward(new_stats["pp_length"], old_stats["pp_length"], np.inf, np.inf),
            "not_arrived_length": get_range_reward(new_stats["not_arrived_length"], old_stats["not_arrived_length"], np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["regions"] * self._rewards["regions"] +\
            rewards["arrived"] * self._rewards["arrived"] +\
            rewards["pp_length"] * self._rewards["pp_length"] +\
            rewards["not_arrived_length"] * self._rewards["not_arrived_length"]

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
        return new_stats["regions"] == 1 and new_stats["arrived"] == True and new_stats["pp_length"] > self._target_length

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
            "regions": new_stats["regions"],
            "pp_length": new_stats["pp_length"],
            "not_arrived_length": new_stats["not_arrived_length"]
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/binary/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/binary/solid.png").convert('RGBA')
            }
        return super().render(map)
