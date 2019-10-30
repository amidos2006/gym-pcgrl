from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
import gym

class PcgrlEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, prob="binary", rep = "narrow"):
        self._prob = PROBLEMS[prob]()
        self._rep = REPRESENTATIONS[rep](self._prob._width, self._prob._height, self._prob._prob)
        self._rep_stats = None

        self.seed()
        self.viewer = None
        self.action_space = self._rep.get_action_space()
        self.observation_space = self._rep.get_observation_space()

    def seed(self, seed=None):
        return [self._rep.seed(seed)]

    def reset(self):
        self._rep.reset()
        self._rep_stats = self._prob.get_stats(self._rep._map)
        return self._rep.get_observation()

    def adjust_param(self, **kwargs):
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)

    def step(self, action):
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        self._rep.update(action)
        self._rep_stats = self._prob.get_stats(self._rep._map)
        #return the values
        return self._rep.get_observation(),self._prob.get_reward(self._rep_stats, old_stats),\
            self._prob.get_episode_over(self._rep_stats,old_stats),self._prob.get_debug_info(self._rep_stats,old_stats)

    def render(self, mode='human'):
        tile_size=16
        img = self._prob.render(self._rep._map)
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
