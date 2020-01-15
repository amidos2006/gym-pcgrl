import os
import numpy as np
from PIL import Image
from gym_pcgrl.envs.probs.binary_prob import BinaryProblem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_longest_path

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class BinaryImpProblem(BinaryProblem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._target_path = 20
        self._random_probs = True

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid"
        target_path (int): the improvement to the path length that the episode turn when it reaches
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
        random_probs (boolean): allow the problem to reset to random uniform probability for initialization of the map
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._random_probs = kwargs.get('random_probs', self._random_probs)

    """
    Resets the problem to the initial state and save the start_stats from the starting map.
    Also, it can be used to change values between different environment resets

    Parameters:
        start_stats (dict(string,any)): the first stats of the map
    """
    def reset(self, start_stats):
        if self._random_probs:
            self._prob["empty"] = self._random.random()
            self._prob["solid"] = 1 - self._prob["empty"]
        
        super().reset(start_stats)

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
        return new_stats["regions"] == 1 and new_stats["path-length"] - self._start_stats["path-length"] >= self._target_path

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
        info = super().get_debug_info(new_stats, old_stats)
        info["path-imp"] = new_stats["path-length"] - self._start_stats["path-length"]
        return info
