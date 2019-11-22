from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gym import spaces
import numpy as np

"""
The wide representation where the agent can pick the tile position and tile value at each update.
"""
class WideRepresentation(Representation):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self):
        super().__init__()

    """
    Gets the action space used by the wide representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        MultiDiscrete: the action space used by that wide representation which
        consists of the x position, y position, and the tile value
    """
    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([width, height, num_tiles])

    """
    Get the observation space used by the wide representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Box: the observation space used by that representation. A 2D array of tile numbers
    """
    def get_observation_space(self, width, height, num_tiles):
        return spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. A 2D array of tile numbers
    """
    def get_observation(self):
        return self._map.copy()

    """
    Get the meaning of all the different actions

    Parameters:
        tiles (string[]): an array of the tile names

    Returns:
        string: that explains the different action names
    """
    def get_action_meaning(self, tiles):
        result  = "For 1st Integer:\n"
        result += "0.." + str(self._map.shape[1]-1) + ": is the x position\n"
        result += "For 2nd Integer:\n"
        result += "0.." + str(self._map.shape[0]-1) + ": is the y position\n"
        result += "For 3rd Integer:\n"
        for i in range(len(tiles)):
            result += str(i) + ": " + tiles[i] + "\n"
        return result

    """
    Get the meaning of the observation

    Parameters:
        tiles (string[]): an array of the tile names

    Returns:
        string: that explains the observation
    """
    def get_observation_meaning(self, tiles):
        result = "The 2D array is the current generated map where the values are:\n"
        for i in range(len(tiles)):
            result += str(i) + ": " + tiles[i] + "\n"
        return result

    """
    Update the wide representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = self._map[action[1]][action[0]] != action[2]
        self._map[action[1]][action[0]] = action[2]
        return change
