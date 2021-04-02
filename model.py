from pdb import set_trace as T
from gym import spaces
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy, MlpExtractor, ActorCriticCnnPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution, CategoricalDistribution, Distribution
import torch as th
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union

conv = th.nn.Conv2d
linear = th.nn.Linear
conv_to_fc = th.nn.Flatten

#TODO: Finish porting models to pytorch
#TODO: Experiment with different architectures, self-attention..?

#def Cnn1(image, **kwargs):
#    activ = tf.nn.relu
#    layer_1 = activ(conv(image, 32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#    layer_3 = conv_to_fc(layer_3)
#
#    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class Cnn1(th.nn.Module):
    def __init__(self, observation_space, **kwargs):
        super().__init__()
        n_chan = observation_space.shape[2]
        self.features_dim = 512
        self.cnn = nn.Sequential(
            conv(n_chan, 32, kernel_size=3, stride=1, **kwargs),
            nn.ReLU(),
            conv(32, 64, kernel_size=3, stride=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, **kwargs),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()).shape[1]
        self.l1 = linear(n_flatten, 512)

    def forward(self, image):
        image = image.permute(0, 3, 1, 2)
        x = self.cnn(image)
        x = self.l1(x)

        return x

class FullCnn(th.nn.Module):
    ''' Like the old FullCnn2, for wide representations with binary or zelda.'''
    def __init__(self, observation_space, n_tools, **kwargs):
        super().__init__()
        self.features_dim = n_tools
        n_chan = observation_space.shape[2]
        act = nn.functional.relu
        self.cnn = nn.Sequential(
            conv(n_chan, 32, kernel_size=3, stride=1, padding=1,  **kwargs),
            nn.ReLU(),
            conv(32, 64, kernel_size=3, stride=1, padding=1,  **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, n_tools, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
        )
        self.act_head = nn.Sequential(
        )

        # Compute shape by doing one forward pass
#       with th.no_grad():
#           _, _, width, height = self.cnn(th.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()).shape
#           assert width == height
#           n_shrink = np.log2(width)
#           assert n_shrink % 1 == 0
#           n_shrink = int(n_shrink)

        self.val_shrink = nn.Sequential(
            conv(n_tools, 64, kernel_size=3, stride=2, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=2, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=1, stride=1, padding=0, **kwargs),
            nn.ReLU(),
        )
        with th.no_grad():
            n_flatten = self.val_shrink(self.cnn(th.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float())).view(-1).shape[0]

        self.val_head = nn.Sequential(
            nn.Flatten(),
            linear(n_flatten, 1)
        )

    def forward(self, image):
        image = image.permute(0, 3, 1, 2)
        x = self.cnn(image)
        act = self.act_head(x)
        val = self.val_head(self.val_shrink(x))

        return act, val

class NCA(th.nn.Module):
    ''' Like the old FullCnn2, for wide representations with binary or zelda.'''
    def __init__(self, observation_space, n_tools, **kwargs):
        super().__init__()
        self.features_dim = n_tools
        n_chan = observation_space.shape[2]
        act = nn.functional.relu
        self.cnn = nn.Sequential(
            conv(n_chan, 32, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(32, 64, kernel_size=3, stride=1, padding=1,  **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, n_tools, kernel_size=3, stride=1, padding=1, **kwargs),
            nn.ReLU(),
        )
        self.act_head = nn.Sequential(
            nn.Flatten(2),
        )

        # Compute shape by doing one forward pass
#       with th.no_grad():
#           _, _, width, height = self.cnn(th.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()).shape
#           assert width == height
#           n_shrink = np.log2(width)
#           assert n_shrink % 1 == 0
#           n_shrink = int(n_shrink)

        self.val_shrink = nn.Sequential(
            conv(n_tools, 64, kernel_size=3, stride=2, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=2, padding=1, **kwargs),
            nn.ReLU(),
            conv(64, 64, kernel_size=1, stride=1, padding=0, **kwargs),
            nn.ReLU(),
        )
        with th.no_grad():
            n_flatten = self.val_shrink(self.cnn(th.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float())).view(-1).shape[0]

        self.val_head = nn.Sequential(
            nn.Flatten(),
            linear(n_flatten, 1)
        )

    def forward(self, image):
        image = image.permute(0, 3, 1, 2)
        x = self.cnn(image)
        act = self.act_head(x)
        act = act.permute(0, 2, 1)
        val = self.val_head(self.val_shrink(x))

        return act, val


class NoDenseMultiCategoricalDistribution(MultiCategoricalDistribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super(NoDenseMultiCategoricalDistribution, self).__init__(action_dims)
        self.action_dims = action_dims
        self.distributions = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

#       action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        action_logits = IdentityModule()
        return action_logits

class NoDenseCategoricalDistribution(CategoricalDistribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super(NoDenseCategoricalDistribution, self).__init__(action_dims)
        self.action_dims = action_dims
        self.distributions = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

#       action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        action_logits = IdentityModule()
        return action_logits

################################################################################
#   Boilerplate policy classes
################################################################################

class CustomPolicyBigMap(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicyBigMap, self).__init__(*args, **kwargs, features_extractor_class=Cnn1)#, feature_extraction="cnn")
import gym

def make_proba_distribution(
    action_space: gym.spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return NoDenseMultiCategoricalDistribution(action_space.nvec)
    elif isinstance(action_space, gym.spaces.Discrete):
        return NoDenseCategoricalDistribution(action_space.n)

class IdentityModule(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = th.nn.Flatten()

    def forward(self, latent_pi):
        return self.flatten(latent_pi)

class CApolicy(ActorCriticCnnPolicy):
    def __init__(self, 
            observation_space,
            action_space,
            lr_schedule,
            **kwargs):
        n_tools = kwargs.pop("n_tools")
        features_extractor_kwargs = {'n_tools': n_tools}
        super(CApolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs, net_arch=None, features_extractor_class=FullCnn, features_extractor_kwargs=features_extractor_kwargs)
        self.action_net = IdentityModule()
        # funky for CA type action
        use_sde = False
        dist_kwargs = None
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        self._build(lr_schedule)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        mean_actions = latent_pi


        if isinstance(self.action_dist, NoDenseMultiCategoricalDistribution) or isinstance(self.action_dist, NoDenseCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        latent_pi, latent_vf = self.extract_features(obs)
       #latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        latent_pi = latent_sde = latent_pi.reshape(latent_pi.shape[0], -1)
        # Evaluate the values for the given observations
#       values = self.value_net(latent_vf)
        values = latent_vf
#       distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        distribution = self.action_dist.proba_distribution(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        latent_pi = latent_sde = latent_pi.reshape(latent_pi.shape[0], -1)
#       distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        distribution = self.action_dist.proba_distribution(latent_pi)
        log_prob = distribution.log_prob(actions)
#       values = self.value_net(latent_vf)
        values = latent_vf
        return values, log_prob, distribution.entropy()




#def FullyConv1(image, n_tools, **kwargs):
#    activ = tf.nn.relu
#    x = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c2', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c3', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c4', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c5', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c6', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c7', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c8', n_filters=n_tools, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    act = conv_to_fc(x)
#    val = activ(conv(x, 'v1', n_filters=64, filter_size=3, stride=2,
#        init_scale=np.sqrt(2)))
#    val = activ(conv(val, 'v4', n_filters=64, filter_size=1, stride=1,
#        init_scale=np.sqrt(2)))
#    val = conv_to_fc(val)
#
#    return act, val
#
#def FullyConv2(image, n_tools, **kwargs):
#    activ = tf.nn.relu
#    x = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c2', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c3', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c4', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c5', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c6', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c7', n_filters=64, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    x = activ(conv(x, 'c8', n_filters=n_tools, filter_size=3, stride=1,
#        pad='SAME', init_scale=np.sqrt(2)))
#    act = conv_to_fc(x)
#    val = activ(conv(x, 'v1', n_filters=64, filter_size=3, stride=2,
#        init_scale=np.sqrt(2)))
#    val = activ(conv(val, 'v2', n_filters=64, filter_size=3, stride=2,
#        init_scale=np.sqrt(3)))
#    val = activ(conv(val, 'v4', n_filters=64, filter_size=1, stride=1,
#        init_scale=np.sqrt(2)))
#    val = conv_to_fc(val)
#
#    return act, val

#class NoDenseCategoricalProbabilityDistributionType(ProbabilityDistributionType):
#    def __init__(self, n_cat):
#        """
#        The probability distribution type for categorical input
#
#        :param n_cat: (int) the number of categories
#        """
#        self.n_cat = n_cat
#
#    def probability_distribution_class(self):
#        return CategoricalProbabilityDistribution
#
#    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0,
#                                       init_bias=0.0):
#        pdparam = pi_latent_vector
#        q_values = vf_latent_vector
#
#        return self.proba_distribution_from_flat(pdparam), pdparam, q_values
#
#    def param_shape(self):
#        return [self.n_cat]
#
#    def sample_shape(self):
#        return []
#
#    def sample_dtype(self):
#        return tf.int64
#
#class FullyConvPolicyBigMap(ActorCriticPolicy):
#    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
#        super(FullyConvPolicyBigMap, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs)
#        n_tools = int(ac_space.n / (ob_space.shape[0] * ob_space.shape[1]))
#        self._pdtype = NoDenseCategoricalProbabilityDistributionType(ac_space.n)
#        with tf.variable_scope("model", reuse=kwargs['reuse']):
#            pi_latent, vf_latent = FullyConv2(self.processed_obs, n_tools, **kwargs)
#            self._value_fn = linear(vf_latent, 'vf', 1)
#            self._proba_distribution, self._policy, self.q_value = \
#                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
#        self._setup_init()
#
#    def step(self, obs, state=None, mask=None, deterministic=False):
#        if deterministic:
#            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
#                                                   {self.obs_ph: obs})
#        else:
#            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
#                                                   {self.obs_ph: obs})
#
#        return action, value, self.initial_state, neglogp
#
#    def proba_step(self, obs, state=None, mask=None):
#        return self.sess.run(self.policy_proba, {self.obs_ph: obs})
#
#    def value(self, obs, state=None, mask=None):
#        return self.sess.run(self.value_flat, {self.obs_ph: obs})
#
#class FullyConvPolicy(ActorCriticPolicy):
#    def __init__(self, **kwargs):
#        super(FullyConvPolicy, self).__init__(**kwargs)
#        ob_space = kwargs.get('observation_space')
#        n_tools = int(ac_space.n / (ob_space.shape[0] * ob_space.shape[1]))
#        self._pdtype = NoDenseCategoricalProbabilityDistributionType(ac_space.n)
#        with tf.variable_scope("model", reuse=kwargs['reuse']):
#            pi_latent, vf_latent = FullyConv1(self.processed_obs, n_tools, **kwargs)
#            self._value_fn = linear(vf_latent, 'vf', 1)
#            self._proba_distribution, self._policy, self.q_value = \
#                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
#        self._setup_init()
#
#    def step(self, obs, state=None, mask=None, deterministic=False):
#        if deterministic:
#            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
#                                                   {self.obs_ph: obs})
#        else:
#            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
#                                                   {self.obs_ph: obs})
#
#        return action, value, self.initial_state, neglogp
#
#    def proba_step(self, obs, state=None, mask=None):
#        return self.sess.run(self.policy_proba, {self.obs_ph: obs})
#
#    def value(self, obs, state=None, mask=None):
#        return self.sess.run(self.value_flat, {self.obs_ph: obs})

#class CustomPolicySmallMap(ActorCriticCnnPolicy):
#    def __init__(self, *args, **kwargs):
#        super(CustomPolicySmallMap, self).__init__(*args, **kwargs, features_extractor_class=Cnn1)#, feature_extraction="cnn")
