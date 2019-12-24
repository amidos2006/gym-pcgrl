from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

"""
Similar to the narrow representation but with ability for not changing anything
or changing 3x3 location at once to a certain value
"""
class NarrowMultiRepresentation(NarrowRepresentation):
    """
    Gets the action space used by the narrow cast representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        MultiDiscrete: the action space used by that narrow multi representation which
        correspond to 9 values. The for all the tiles in a 3x3 grid
    """
    def get_action_space(self, width, height, num_tiles):
        action_space = []
        for i in range(9):
            action_space.append(num_tiles + 1)
        return spaces.MultiDiscrete(action_space)

    """
    Update the narrow cast representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = 0
        low_y,high_y=max(self._y-1,0),min(self._y+2,self._map.shape[0])
        low_x,high_x=max(self._x-1,0),min(self._x+2,self._map.shape[1])
        for i in range(len(action)):
            x, y = self._x + (i % 3)-1, self._y + int(i / 3)-1
            if x >= low_x and x < high_x and y >= low_y and y < high_y and action[i] > 0:
                change += [0,1][self._map[y][x] != action[i]-1]
                self._map[y][x] = action[i]-1

        if self._random_tile:
            self._x = self._random.randint(self._map.shape[1])
            self._y = self._random.randint(self._map.shape[0])
        else:
            self._x += 1
            if self._x >= self._map.shape[1]:
                self._x = 0
                self._y += 1
                if self._y >= self._map.shape[0]:
                    self._y = 0
        return change, self._x, self._y
