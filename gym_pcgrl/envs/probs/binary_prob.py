import os
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.probs.helper import calc_num_regions, calc_longest_path

class BinaryProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 14
        self._height = 14
        self._prob = {"empty": 0.7, "solid":0.3}

        self._target_path = 50

        self._rewards = {
            "regions": 5,
            "path-length": 1
        }

    def get_tile_types(self):
        return ["empty", "solid"]

    def adjust_param(self, **kwargs):
        self._width, self._height = kwargs.get('width', self._width), kwargs.get('height', self._height)
        self._prob["empty"] = kwargs.get('empty_prob', self._prob["empty"])
        self._prob["solid"] = kwargs.get('solid_prob', self._prob["solid"])

        self._target_path = kwargs.get('target_path', self._target_path)

        self._rewards = {
            "regions": kwargs.get('reward_regions', self._rewards["regions"]),
            "path-length": kwargs.get('reward_path_length', self._rewards["path-length"])
        }

    def get_stats(self, map):
        return {
            "regions": calc_num_regions(map, ["empty"]),
            "path-length": calc_longest_path(map, ["empty"])
        }

    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "regions": old_stats["regions"] - new_stats["regions"],
            "path-length": new_stats["path-length"] - old_stats["path-length"]
        }
        #unless the number of regions become zero, it has to be punished
        if new_stats["regions"] == 0 and old_stats["regions"] > 0:
            rewards["regions"] = -1
        #calculate the total reward
        return rewards["regions"] * self._rewards["regions"] +\
            rewards["path-length"] * self._rewards["path-length"]

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["regions"] == 1 and new_stats["path-length"] >= self._target_path

    def get_debug_info(self, new_stats, old_stats):
        return {
            "regions": new_stats["regions"],
            "path-length": new_stats["path-length"]
        }

    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/binary/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/binary/solid.png").convert('RGBA')
            }
        return super().render(map)
