import tensorflow as tf
from tensorflow.keras import layers
from gym import spaces
import numpy as np
from stable_baselines.common.policies import ActorCriticPolicy, FeedForwardPolicy
from stable_baselines.common.distributions import CategoricalProbabilityDistributionType, ProbabilityDistributionType, CategoricalProbabilityDistribution, ProbabilityDistribution
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

def Cnn1(image, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

def Cnn2(image, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

def FullyConv1(image, n_tools, **kwargs):
    activ = tf.nn.relu
    x = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c2', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c3', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c4', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c5', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c6', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c7', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c8', n_filters=n_tools, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    act = conv_to_fc(x)
    val = activ(conv(x, 'v1', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(2)))
    val = activ(conv(val, 'v4', n_filters=64, filter_size=1, stride=1,
        init_scale=np.sqrt(2)))
    val = conv_to_fc(val)
    return act, val

def FullyConv2(image, n_tools, **kwargs):
    activ = tf.nn.relu
    x = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c2', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c3', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c4', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c5', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c6', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c7', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c8', n_filters=n_tools, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    act = conv_to_fc(x)
    val = activ(conv(x, 'v1', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(2)))
    val = activ(conv(val, 'v2', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(3)))
    val = activ(conv(val, 'v4', n_filters=64, filter_size=1, stride=1,
        init_scale=np.sqrt(2)))
    val = conv_to_fc(val)
    return act, val

class NoDenseCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_cat):
        """
        The probability distribution type for categorical input

        :param n_cat: (int) the number of categories
        """
        self.n_cat = n_cat

    def probability_distribution_class(self):
        return CategoricalProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0,
                                       init_bias=0.0):
        pdparam = pi_latent_vector
        q_values = vf_latent_vector
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.n_cat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int64

class FullyConvPolicyBigMap(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
        super(FullyConvPolicyBigMap, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs)
        n_tools = int(ac_space.n / (ob_space.shape[0] * ob_space.shape[1]))
        self._pdtype = NoDenseCategoricalProbabilityDistributionType(ac_space.n)
        with tf.variable_scope("model", reuse=kwargs['reuse']):
            pi_latent, vf_latent = FullyConv2(self.processed_obs, n_tools, **kwargs)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

class FullyConvPolicySmallMap(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
        super(FullyConvPolicySmallMap, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs)
        n_tools = int(ac_space.n / (ob_space.shape[0] * ob_space.shape[1]))
        self._pdtype = NoDenseCategoricalProbabilityDistributionType(ac_space.n)
        with tf.variable_scope("model", reuse=kwargs['reuse']):
            pi_latent, vf_latent = FullyConv1(self.processed_obs, n_tools, **kwargs)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

class CustomPolicyBigMap(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicyBigMap, self).__init__(*args, **kwargs, cnn_extractor=Cnn2, feature_extraction="cnn")

class CustomPolicySmallMap(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicySmallMap, self).__init__(*args, **kwargs, cnn_extractor=Cnn1, feature_extraction="cnn")
