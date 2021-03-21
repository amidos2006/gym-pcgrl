import gym
import gym_pcgrl

import numpy as np
import math
import os

# clean the input action
get_action = lambda a: a.item() if hasattr(a, "item") else a
# unwrap all the environments and get the PcgrlEnv
get_pcgrl_env = lambda env: env if "PcgrlEnv" in str(type(env)) else get_pcgrl_env(env.env)

"""
Return a Box instead of dictionary by stacking different similar objects

Can be stacked as Last Layer
"""
class ToImage(gym.Wrapper):
    def __init__(self, game, names, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
        self.shape = None
        depth=0
        max_value = 0
        for n in names:
            assert n in self.env.observation_space.spaces.keys(), 'This wrapper only works if your observation_space is spaces.Dict with the input names.'
            if self.shape == None:
                self.shape = self.env.observation_space[n].shape
            new_shape = self.env.observation_space[n].shape
            depth += 1 if len(new_shape) <= 2 else new_shape[2]
            assert self.shape[0] == new_shape[0] and self.shape[1] == new_shape[1], 'This wrapper only works when all objects have same width and height'
            if self.env.observation_space[n].high.max() > max_value:
                max_value = self.env.observation_space[n].high.max()
        self.names = names

        self.observation_space = gym.spaces.Box(low=0, high=max_value,shape=(self.shape[0], self.shape[1], depth))

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        final = np.empty([])
        for n in self.names:
            if len(final.shape) == 0:
                final = obs[n].reshape(self.shape[0], self.shape[1], -1)
            else:
                final = np.append(final, obs[n].reshape(self.shape[0], self.shape[1], -1), axis=2)
        return final

"""
Transform any object in the dictionary to one hot encoding

can be stacked
"""
class OneHotEncoding(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a {} key'.format(name)
        self.name = name

        self.observation_space = gym.spaces.Dict({})
        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        new_shape = []
        shape = self.env.observation_space[self.name].shape
        self.dim = self.observation_space[self.name].high.max() - self.observation_space[self.name].low.min() + 1
        for v in shape:
            new_shape.append(v)
        new_shape.append(self.dim)
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.uint8)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        old = obs[self.name]
        obs[self.name] = np.eye(self.dim)[old]
        return obs

"""
Transform the input space to a 3D map of values where the argmax value will be applied

can be stacked
"""
class ActionMap(gym.Wrapper):
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'map' in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a map key'
        self.old_obs = None
        self.one_hot = len(self.env.observation_space['map'].shape) > 2
        w, h, dim = 0, 0, 0
        if self.one_hot:
            h, w, dim = self.env.observation_space['map'].shape
        else:
            h, w = self.env.observation_space['map'].shape
            dim = self.env.observation_space['map'].high.max()
        self.h = self.unwrapped.h = h
        self.w = self.unwrapped.w = w
        self.dim = self.unwrapped.dim = self.env.get_num_tiles()
       #self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(h,w,dim))
        self.action_space = gym.spaces.Discrete(h*w*self.dim)

    def reset(self):
        self.old_obs = self.env.reset()
        return self.old_obs

    def step(self, action):
       #y, x, v = np.unravel_index(np.argmax(action), action.shape)
        y, x, v = np.unravel_index(action, (self.h, self.w, self.dim))
        if 'pos' in self.old_obs:
            o_x, o_y = self.old_obs['pos']
            if o_x == x and o_y == y:
                obs, reward, done, info = self.env.step(v)
            else:
                o_v = self.old_obs['map'][o_y][o_x]
                if self.one_hot:
                    o_v = o_v.argmax()
                obs, reward, done, info = self.env.step(o_v)
        else:
            obs, reward, done, info = self.env.step([x, y, v])
        self.old_obs = obs
        return obs, reward, done, info

"""
Crops and centers the view around the agent and replace the map with cropped version
The crop size can be larger than the actual view, it just pads the outside
This wrapper only works on games with a position coordinate

can be stacked
"""
class Cropped(gym.Wrapper):
    def __init__(self, game, crop_size, pad_value, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'pos' in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a position'
        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a {} key'.format(name)
        assert len(self.env.observation_space.spaces[name].shape) == 2, "This wrapper only works on 2D arrays."
        self.name = name
        self.size = crop_size
        self.pad = crop_size//2
        self.pad_value = pad_value

        self.observation_space = gym.spaces.Dict({})
        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        high_value = self.observation_space[self.name].high.max()
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=high_value, shape=(crop_size, crop_size), dtype=np.uint8)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        map = obs[self.name]
        x, y = obs['pos']

        #View Centering
        padded = np.pad(map, self.pad, constant_values=self.pad_value)
        cropped = padded[y:y+self.size, x:x+self.size]
        obs[self.name] = cropped

        return obs

################################################################################
#   Final used wrappers for the experiments
################################################################################

"""
The wrappers we use for narrow and turtle experiments
"""
class CroppedImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, crop_size, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Cropping the map to the correct crop_size
        env = Cropped(self.pcgrl_env, crop_size, self.pcgrl_env.get_border_tile(), 'map')
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Indices for flatting
        flat_indices = ['map']
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)

"""
Similar to the previous wrapper but the input now is the index in a 3D map (height, width, num_tiles) of the highest value
Used for wide experiments
"""
class ActionMapImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Indices for flatting
        flat_indices = ['map']
        env = self.pcgrl_env
        # Add the action map wrapper
        env = ActionMap(env)
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)

################################################################################
#   Conditional Wrapper
################################################################################

class ParamRew(gym.Wrapper):
    def __init__(self, env, env_params, rand_params=False):

        # Whether to always select random parameters, to stabilize learning multiple objectives 
        # (i.e. prevent overfitting to some subset of objectives)
        self.rand_params = rand_params
        self.env = env
        super().__init__(self.env)
        self.usable_metrics = env_params
        self.num_params = len(env_params)
        self.auto_reset = True

    def configure(self, **kwargs):
        print(kwargs)
        self.unwrapped.configure(**kwargs)
        self.metrics = self.unwrapped.metrics
        # NB: self.metrics needs to be an OrderedDict
        print('usable metrcs:', self.usable_metrics)
        print('unwrapped: {}'.format(self.unwrapped.metrics))
        self.last_metrics = copy.deepcopy(self.metrics)
        self.param_bounds = self.unwrapped.param_bounds
        self.param_ranges = {}
#       self.max_improvement = 0
        for k in self.usable_metrics:
            v = self.param_bounds[k]
            improvement = abs(v[1] - v[0])
            self.param_ranges[k] = improvement
#           self.max_improvement += improvement * self.weights[k]
        self.metric_trgs = self.unwrapped.metric_trgs

        for k in self.usable_metrics:
            v = self.metrics[k]
            self.weights[k] = self.unwrapped.weights[k]
        for k in self.usable_metrics:
            self.param_bounds['{}_weight'.format(k)] = (0, 1)
        self.width = self.unwrapped.width
        self.observation_space = self.env.observation_space
        # FIXME: hack for gym-pcgrl
        print('param orig rew obs space', self.observation_space)
        self.action_space = self.env.action_space
        orig_obs_shape = self.observation_space.shape
        obs_shape = orig_obs_shape[0] + 2 * len(self.usable_metrics), orig_obs_shape[1], orig_obs_shape[2]
        low = self.observation_space.low
        high = self.observation_space.high
        metrics_shape = (2 * len(self.usable_metrics), obs_shape[1], obs_shape[2])
        self.metrics_shape = metrics_shape
        metrics_low = np.full(metrics_shape, fill_value=0)
        metrics_high = np.full(metrics_shape, fill_value=1)
        low = np.vstack((metrics_low, low))
        high = np.vstack((metrics_high, high))
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.next_trgs = None
        if self.render_gui and True:
            screen_width = 200
            screen_height = 100 * self.num_params
            from wrapper_windows import ParamRewWindow
            win = ParamRewWindow(self, self.metrics, self.metric_trgs, self.param_bounds)
           #win.connect("destroy", Gtk.main_quit)
            win.show_all()
            self.win = win

    def enable_auto_reset(self, button):
        self.auto_reset = button.get_active()


    def getState(self):
        scalars = super().getState()
       #trg_weights = [v for k, v in self.weights.items()]
       #scalars += trg_weights
        print(scalars)
        raise Exception
        return scalars

    def set_trgs(self, trgs):
        if self.rand_params:
            for k in self.usable_metrics:
                min_v, max_v = self.param_bounds[k]
                trgs[k] = random.uniform(min_v, max_v)
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
        metrics_ob = np.zeros((self.metrics_shape))
        i = 0
        for k in self.usable_metrics:
            trg = self.metric_trgs[k]
            metric = self.metrics[k]
            if not metric:
                #FIXME: a problem after reset in pcgrl envs
#               print(k, metric, self.metrics)
#               assert self.n_step < 20
                metric = 0
            metrics_ob[i*2, :, :] = trg / self.param_ranges[k]
            metrics_ob[i*2+1, :, :] = metric / self.param_ranges[k]
            i += 1
#       print('param rew obs shape ', obs.shape)
#       print('metric trgs shape ', metrics_ob.shape)
        obs = np.vstack((metrics_ob, obs))
        return obs


    def step(self, action):
        if self.render_gui and True:
            self.win.step()

       #print('unwrapped metrics', self.unwrapped.metrics)
        ob, rew, done, info = super().step(action)
        ob = self.observe_metric_trgs(ob)
        self.metrics = self.unwrapped.metrics
        rew = self.get_reward()
        self.last_metrics = copy.deepcopy(self.metrics)
        self.n_step += 1
        if not self.auto_reset:
            done = False
        return ob, rew, done, info

    def get_param_trgs(self):
        return self.metric_trgs

    def get_param_bounds(self):
        return self.param_bounds

    def set_param_bounds(self, bounds):
        for k, (l, h) in bounds.items():
            self.param_bounds[k] = (l, h)

    def display_metric_trgs(self):
        if self.render_gui:
            self.win.display_metric_trgs()

    def get_reward(self):
        reward = 0

        for metric in self.usable_metrics:
            trg = self.metric_trgs[metric]
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


#FIXME: sometimes the mere existence of this class will break a multi-env micropolis run
class ParamRewWindow(Gtk.Window):
    def __init__(self, env, metrics, metric_trgs, metric_bounds):
        self.env = env
        import gi 
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk, GLib
        Gtk.Window.__init__(self, title="Metrics")
        self.set_border_width(10)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        
        reset_button = Gtk.Button("reset")
        reset_button.connect('clicked', lambda item: self.env.reset())
        hbox.pack_start(reset_button, False, False, 0) 

        auto_reset_button = Gtk.CheckButton("auto reset")
        auto_reset_button.connect('clicked', lambda item: self.env.enable_auto_reset(item))
        auto_reset_button.set_active(True)
        hbox.pack_start(auto_reset_button, False, False, 0)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox.pack_start(hbox, False, False, 0)
        self.add(vbox)

        prog_bars = {}
        scales = {}
        prog_labels = {}
        for k in metrics:
            if k not in self.env.usable_metrics:
                continue
            metric = metrics[k]
            label = Gtk.Label()
            label.set_text(k)
            vbox.pack_start(label, True, True, 0)
            if metric is None:
                metric = 0
            ad = Gtk.Adjustment(metric, metric_bounds[k][0], metric_bounds[k][1],
                                env.param_ranges[k] / 20, env.param_ranges[k] / 10, 0)
            scale = Gtk.HScale(adjustment=ad)
            scale.set_name(k)
            scale.set_show_fill_level(True)
            scales[k] = scale
            vbox.pack_start(scale, True, True, 0)
            scale.connect("value-changed", self.scale_moved)

            prog_label = Gtk.Label()
            prog_label.set_text(str(metric))
            prog_labels[k] = prog_label
            vbox.pack_start(prog_label, True, True, 0)
            metric_prog = Gtk.ProgressBar()
#           metric_prog.set_draw_value(True)
            prog_bars[k] = metric_prog
            vbox.pack_start(metric_prog, True, True, 10)
           #bounds = metric_bounds[k]
           #frac = metrics[k]
           #metric_prog.set_fraction(frac)

      
       #self.timeout_id = GLib.timeout_add(50, self.on_timeout, None)
       #self.activity_mode = False
        self.prog_bars = prog_bars
        self.scales = scales
        self.prog_labels = prog_labels



    def step(self):
        self.display_metrics()
        while Gtk.events_pending():
            Gtk.main_iteration()

    def scale_moved(self, event):
        k = event.get_name()
        self.env.metric_trgs[k] = event.get_value()
        self.env.set_trgs(self.env.metric_trgs)

    def display_metric_trgs(self):
        for k, v in self.env.metric_trgs.items():
            if k in self.env.usable_metrics:
                self.scales[k].set_value(v)

    def display_metrics(self):
        for k, prog_bar in self.prog_bars.items():
            metric_val = self.env.metrics[k]
            prog_bar.set_fraction(metric_val / self.env.param_ranges[k])
            prog_label = self.prog_labels[k]
            prog_label.set_text(str(metric_val))

    def on_show_text_toggled(self, button):
        show_text = button.get_active()
        if show_text:
            text = "some text"
        else:
            text = None
        self.progressbar.set_text(text)
        self.progressbar.set_show_text(show_text)

    def on_activity_mode_toggled(self, button):
        self.activity_mode = button.get_active()
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            self.progressbar.set_fraction(0.0)

    def on_right_to_left_toggled(self, button):
        value = button.get_active()
        self.progressbar.set_inverted(value)

    def on_timeout(self, user_data):
        """
        Update value on the progress bar
        """
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            new_value = self.progressbar.get_fraction() + 0.01

            if new_value > 1:
                new_value = 0

            self.progressbar.set_fraction(new_value)

        # As this is a timeout function, return True so that it
        # continues to get called
        return True

