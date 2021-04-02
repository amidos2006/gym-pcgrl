################################################################################
#   Conditional Wrapper
################################################################################
from pdb import set_trace as T
import copy
import gym
from opensimplex import OpenSimplex
import numpy as np

class ParamRew(gym.Wrapper):
    def __init__(self, env, cond_metrics, rand_params=False, **kwargs):

        self.render_gui = kwargs.get('render')
        # Whether to always select random parameters, to stabilize learning multiple objectives
        # (i.e. prevent overfitting to some subset of objectives)
#       self.rand_params = rand_params
        self.env = env
        super().__init__(self.env)
#       cond_trgs = self.unwrapped.cond_trgs
        self.usable_metrics = cond_metrics
        self.static_metrics = list(env.static_trgs.keys())
        self.num_params = len(self.usable_metrics)
        self.auto_reset = True
        self.weights = {}

#       self.unwrapped.configure(**kwargs)
        self.metrics = self.unwrapped.metrics
        # NB: self.metrics needs to be an OrderedDict
        print('usable metrics for conditional wrapper:', self.usable_metrics)
        print('unwrapped env\'s current metrics: {}'.format(self.unwrapped.metrics))
        self.last_metrics = copy.deepcopy(self.metrics)
        self.cond_bounds = self.unwrapped.cond_bounds
        self.param_ranges = {}
#       self.max_improvement = 0
        for k in self.usable_metrics:
            v = self.cond_bounds[k]
            improvement = abs(v[1] - v[0])
            self.param_ranges[k] = improvement
#           self.max_improvement += improvement * self.weights[k]

        self.metric_trgs = {}
        # we might be using a subset of possible conditional targets supplied by the problem
        for k in self.usable_metrics:
            self.metric_trgs[k] = self.unwrapped.cond_trgs[k]

        for k in self.usable_metrics + self.static_metrics:
            v = self.metrics[k]
            self.weights[k] = self.unwrapped.weights[k]

        for k in self.usable_metrics:
            self.cond_bounds['{}_weight'.format(k)] = (0, 1)
        self.width = self.unwrapped.width
        self.observation_space = self.env.observation_space
        # FIXME: hack for gym-pcgrl
        print('conditional wrapper, original observation space', self.observation_space)
        self.action_space = self.env.action_space
        orig_obs_shape = self.observation_space.shape
        #TODO: adapt to (c, w, h) vs (w, h, c)
        n_new_obs = 1 * len(self.usable_metrics)
        obs_shape = orig_obs_shape[0], orig_obs_shape[1], orig_obs_shape[2] + n_new_obs
        low = self.observation_space.low
        high = self.observation_space.high
        metrics_shape = (obs_shape[0], obs_shape[1], n_new_obs)
        self.metrics_shape = metrics_shape
        metrics_low = np.full(metrics_shape, fill_value=0)
        metrics_high = np.full(metrics_shape, fill_value=1)
        low = np.concatenate((metrics_low, low), axis=2)
        high = np.concatenate((metrics_high, high), axis=2)
        self.observation_space = gym.spaces.Box(low=low, high=high)
        # Yikes lol (this is to appease SB3)
        self.unwrapped.observation_space = self.observation_space
        print('conditional observation space: {}'.format(self.observation_space))
        self.next_trgs = None

        if self.render_gui and True:
            screen_width = 200
            screen_height = 100 * self.num_params
            from gym_pcgrl.conditional_window import ParamRewWindow
            win = ParamRewWindow(self, self.metrics, self.metric_trgs, self.cond_bounds)
           #win.connect("destroy", Gtk.main_quit)
            win.show_all()
            self.win = win
        self.infer = kwargs.get('infer', False)

    def configure(self, **kwargs):
        pass



    def enable_auto_reset(self, button):
        self.auto_reset = button.get_active()


    def getState(self):
        scalars = super().getState()
       #trg_weights = [v for k, v in self.weights.items()]
       #scalars += trg_weights
        print('scalars: ', scalars)
        raise Exception

        return scalars

    def set_trgs(self, trgs):
#       if self.rand_params:
#           for k in self.usable_metrics:
#               min_v, max_v = self.cond_bounds[k]
#               trgs[k] = random.uniform(min_v, max_v)
        self.next_trgs = trgs

    def do_set_trgs(self):
        trgs = self.next_trgs
        i = 0
        self.init_metrics = copy.deepcopy(self.metrics)

#       self.max_improvement = 0
        for k, trg in trgs.items():
            if k in self.usable_metrics:
                self.metric_trgs[k] = trg
#               self.max_improvement += abs(trg - self.init_metrics[k]) * self.weights[k]

       #for k in self.usable_metrics:
       #    weight_name = '{}_weight'.format(k)
       #    if weight_name in params:
       #        self.weights[k] = params[weight_name]
       #    i += 1

       #for k, _ in self.usable_metrics():
       #    if k in params:
       #        metric_trg = params[k]
       #        self.metric_trgs[k] = metric_trg
       #        self.max_improvement += abs(metric_trg - self.init_metrics[k]) * self.weights[k]
       #    i += 1


#       print('set trgs {}'.format(self.metric_trgs))
        self.display_metric_trgs()

    def reset(self):
        if self.next_trgs:
            self.do_set_trgs()
        ob = super().reset()
        ob = self.observe_metric_trgs(ob)
        self.metrics = self.unwrapped.metrics
        self.last_metrics = copy.deepcopy(self.metrics)
        self.n_step = 0

        return ob

    def observe_metric_trgs(self, obs):
        metrics_ob = np.zeros(self.metrics_shape)
        i = 0

        for k in self.usable_metrics:
            if k in self.static_trgs:
                continue
            trg = self.metric_trgs[k]
            metric = self.metrics[k]

            if not metric:
                #FIXME: a problem after reset in pcgrl envs
#               print(k, metric, self.metrics)
#               assert self.n_step < 20
                metric = 0
            trg_range = self.param_ranges[k]
            metrics_ob[:, :, i] = np.sign(trg / trg_range - metric / trg_range)
#           metrics_ob[:, :, i*2] = trg / self.param_ranges[k]
#           metrics_ob[:, :, i*2+1] = metric / self.param_ranges[k]
            i += 1
#       print('param rew obs shape ', obs.shape)
#       print('metric trgs shape ', metrics_ob.shape)
        obs = np.concatenate((metrics_ob, obs), axis=2)

        return obs


    def step(self, action):
        if self.render_gui and True:
            self.win.step()

        ob, rew, done, info = super().step(action)
        ob = self.observe_metric_trgs(ob)
        self.metrics = self.unwrapped.metrics
        rew = self.get_reward()
        self.last_metrics = copy.deepcopy(self.metrics)
        self.n_step += 1

        if self.auto_reset:
            # either exceeded number of changes, steps, or have reached target
            done = done or self.get_done()
        else:
            assert self.infer
            done = False
        return ob, rew, done, info

    def get_cond_trgs(self):
        return self.metric_trgs

    def get_cond_bounds(self):
        return self.cond_bounds

    def set_cond_bounds(self, bounds):
        for k, (l, h) in bounds.items():
            self.cond_bounds[k] = (l, h)

    def display_metric_trgs(self):
        if self.render_gui:
            self.win.display_metric_trgs()

    def get_reward(self):
        reward = 0

        for metric in self.usable_metrics + list(self.static_trgs.keys()):
            if metric in self.metric_trgs:
                trg = self.metric_trgs[metric]
            elif metric in self.static_trgs:
                trg = self.static_trgs[metric]
            val = self.metrics[metric]
            last_val = self.last_metrics[metric]
            trg_change = trg - last_val
            change = val - last_val
            metric_rew = 0
            same_sign = (change < 0) == (trg_change < 0)
            # changed in wrong direction

            if not same_sign:
                metric_rew -= abs(change)
            else:
                less_change = abs(change) < abs(trg_change)
                # changed not too much, in the right direction

                if less_change:
                    metric_rew += abs(change)
                else:
                    metric_rew += abs(trg_change) - abs(trg_change - change)
            reward += metric_rew * self.weights[metric]

       #assert(reward <= self.max_improvement, 'actual reward {} is less than supposed maximum possible \
       #        improvement toward target vectors of {}'.format(reward, self.max_improvement))
       #if self.max_improvement == 0:
       #    pass
       #else:
       #    reward = 100 * (reward / self.max_improvement)

        return reward

    def get_done(self):
        done = True
        trg_dict = self.metric_trgs
        trg_dict.update(self.static_trgs)
        for k, v in trg_dict.items():
            if self.metrics[k] != int(v):
                done = False
        if done and self.infer:
            print('targets reached! {}'.format(trg_dict))
        return done

# TODO: What the fuck is this actually doing and why does it kind of work?
class PerlinNoiseyTargets(gym.Wrapper):
    '''A bunch of simplex noise instances modulate target metrics.'''
    def __init__(self, env, **kwargs):
        super(PerlinNoiseyTargets, self).__init__(env)
        self.cond_bounds = self.env.unwrapped.cond_bounds
        self.num_params = self.num_params
        self.noise = OpenSimplex()
        # Do not reset n_step so that we keep moving through the perlin noise and do not repeat our course
        self.n_step = 0
        self.X, self.Y = np.random.random(2) * 10000

    def step(self, a):
        cond_bounds = self.cond_bounds
        trgs = {}
        i = 0

        for k in self.env.usable_metrics:
            trgs[k] = self.noise.noise2d(x=self.X + self.n_step/400, y=self.Y + i*100)
            i += 1

        i = 0

        for k in self.env.usable_metrics:
            (ub, lb) = cond_bounds[k]
            trgs[k] = ((trgs[k] + 1) / 2 * (ub - lb)) + lb
            i += 1
        self.env.set_trgs(trgs)
        out = self.env.step(a)
        self.n_step += 1

        return out

    def reset(self):
#       self.noise = OpenSimplex()
        return self.env.reset()

class UniformNoiseyTargets(gym.Wrapper):
    '''A bunch of simplex noise instances modulate target metrics.'''
    def __init__(self, env, **kwargs):
        super(UniformNoiseyTargets, self).__init__(env)
        self.cond_bounds = self.env.unwrapped.cond_bounds
        self.num_params = self.num_params
        self.midep_trgs = kwargs.get('midep_trgs', False)

    def set_rand_trgs(self):
        trgs = {}

        for k in self.env.usable_metrics:
            (lb, ub) = self.cond_bounds[k]
            trgs[k] = np.random.random() * (ub - lb) + lb
        self.env.set_trgs(trgs)

    def step(self, action):
        if self.midep_trgs:
            if np.random.random() < 0.005:
                self.set_rand_trgs()
                self.do_set_trgs()

        return self.env.step(action)


    def reset(self):
        self.set_rand_trgs()

        return self.env.reset()
