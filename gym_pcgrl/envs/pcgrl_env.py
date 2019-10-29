from gym_pcgrl.envs.reps import NarrowRepresentation, WideRepresentation, TurtleRepresentation
import gym

class PcgrlEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, rep = "narrow"):
        if rep == "narrow":
            self._rep = NarrowRepresentation()
        elif rep == "wide":
            self._rep = WideRepresentation()
        elif rep == "turtle":
            self._rep = TurtleRepresentation()
        self._rep_stats = None

        self.seed()
        self._init_param()
        self.viewer = None
        self.action_space = self._rep.get_action_space()
        self.observation_space = self._rep.get_observation_space()

    def seed(self, seed=None):
        return [self._rep.seed(seed)]

    def _calc_rep_stats(self):
        raise NotImplementedError('_calc_rep_stats is not implemeneted.')

    def reset(self):
        self._rep.reset()
        self._calc_rep_stats()
        return self._rep.get_observation()

    def _init_param(self):
        raise NotImplementedError('_init_param is not implemeneted.')

    def adjust_param(self, **kwargs):
        self._rep.adjust_param(**kwargs)

    def _calc_total_reward(self, old_stats):
        raise NotImplementedError('_calc_total_reward is not implemeneted.')

    def _calc_episode_over(self, old_stats):
        raise NotImplementedError('_calc_episode_over is not implemeneted.')

    def _calc_debug_info(self, old_stats):
        raise NotImplementedError('_calc_debug_info is not implemeneted.')

    def step(self, action):
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        self._rep.update(action)
        self._calc_rep_stats()
        #return the values
        return self._rep.get_observation(),self._calc_total_reward(old_stats),self._calc_episode_over(old_stats),self._calc_debug_info(old_stats)

    def render(self, graphics, padding_tile, tile_size=16, mode='human'):
        img = self._rep.render(graphics, padding_tile, tile_size).convert("RGB")
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
