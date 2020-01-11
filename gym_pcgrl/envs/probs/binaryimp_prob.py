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

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action
        start_stats (dict(string,any)): the first stats of the map

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, old_stats, start_stats):
        return new_stats["regions"] == 1 and new_stats["path-length"] - start_stats["path-length"] >= self._target_path

    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action
        start_stats (dict(string,any)): the first stats of the map

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats, start_stats):
        info = super().get_debug_info(new_stats, old_stats, start_stats)
        info["path-imp"] = new_stats["path-length"] - start_stats["path-length"]
        return info
