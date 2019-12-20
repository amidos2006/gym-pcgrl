from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

"""
Similar to the narrow representation but with ability for not changing anything
or changing 3x3 location at once to a certain value
"""
class NarrowCastRepresentation(NarrowRepresentation):
    """
    Gets the action space used by the narrow cast representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        MultiDiscrete: the action space used by that narrow cast representation which
        correspond to two values. The first is the type of action and the second is the tile type
    """
    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([3, num_tiles])

    """
    Update the narrow cast representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        type, value = action
        change = 0
        if type ==1 :
            change += [0,1][self._map[self._y][self._x] != value]
            self._map[self._y][self._x] = value
        elif type == 2:
            low_y,high_y=max(self._y-1,0),min(self._y+2,self._map.shape[0])
            low_x,high_x=max(self._x-1,0),min(self._x+2,self._map.shape[1])
            for y in range(low_y,high_y):
                for x in range(low_x,high_x):
                    change += [0,1][self._map[y][x] != value]
                    self._map[y][x] = value
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
