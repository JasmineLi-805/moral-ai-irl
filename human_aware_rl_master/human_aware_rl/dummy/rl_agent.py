import os, sys
from typing import Dict

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from ray.rllib.models.preprocessors import Preprocessor
sys.path.append(os.path.dirname('/Users/jasmineli/Desktop/moral-ai-irl/overcooked_demo_litw'))
from overcooked_demo_litw.server.game import MAIDumbAgent, MAIDumbAgentLeftCoop, MAIDumbAgentRightCoop
from overcooked_ai_py.mdp.actions import Action
from ray.rllib.policy import Policy as RllibPolicy
import numpy as np
from tensorflow.compat.v1.keras.backend import set_session, get_session
from collections import OrderedDict
from gym.spaces import Space
from gym.utils import seeding


class DummyPolicy(RllibPolicy):
    """
    This wraps the preprogrammed Dummy Policy into an RllibPolicy
    """
    def __init__(self, observation_space, action_space, config):
        super(DummyPolicy, self).__init__(observation_space, action_space, config)

        assert config
        assert config['layout']
        
        possible_layout = ['mai_separate_coop_left', 'mai_separate_coop_right']
        assert config['layout'] in possible_layout
        self.model = MAIDumbAgentLeftCoop() if config['layout'] == 'mai_separate_coop_left' else MAIDumbAgentRightCoop()
        # self.context = self._create_execution_context()

        print('a DummyPolicy is instantiated')

    # def _setup_shapes(self):
    #     # This is here to make the class compatible with both tuples or gym.Space objs for the spaces
    #     # Note: action_space = (len(Action.ALL_ACTIONS,)) is technically NOT the action space shape, which would be () since actions are scalars
    #     self.observation_shape = self.observation_space if type(self.observation_space) == tuple else self.observation_space.shape
    #     self.action_shape = self.action_space if type(self.action_space) == tuple else (self.action_space.n,)

    def compute_actions(self, obs_batch, 
                        state_batches=None, 
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """
        Computes sampled actions for each of the corresponding OvercookedEnv states in obs_batch

        Args:
            obs_batch (np.array): batch of pre-process (lossless state encoded) observations

        Returns:
            actions (list|np.array): batch of output actions shape [BATCH_SIZE, ACTION_SHAPE]
            rnn_state: if using lstm models
            info (dict)
        """
        actions = []
        for obs in obs_batch:
            actions.append(Action.ACTION_TO_INDEX[self.model.action(obs)[0]])
        return actions, [], {}

    def get_initial_state(self):
        return []


    def get_weights(self):
        """
        No-op to keep rllib from breaking, won't be necessary in future rllib releases
        """
        pass

    def set_weights(self, weights):
        """
        No-op to keep rllib from breaking
        """
        pass


    def learn_on_batch(self, samples):
        """
        Static policy requires no learning
        """
        return {}

    def _forward(self, obs_batch, state_batches):
        pass

    def _create_execution_context(self):
        pass


class DummyObservationSpace(Space):
    """
    The Dummy agent takes the game state dictionary as its input. This class is a wrapper
    for the game state so that the OvercookedState object is compatible with Rllib.

    Example usage:
    self.observation_space = gym.spaces.DummyObservationSpace()
    """

    def __init__(self, spaces=None, **spaces_kwargs):
        self.spaces = spaces
        self.shape = -1
        # super(DummyObservationSpace, self).__init__(
        #     self.shape, None
        # )  # None for shape and dtype, since it'll require special handling

    def contains(self, x):
        if isinstance(x, OvercookedState):
            x = x.to_dict()
        elif not isinstance(x, Dict):
            return False

        keys = ['players', 'objects', 'bonus_orders', 'all_orders', 'timestep']
        for k in keys:
            if k not in x:
                return False
        return True

    def __getitem__(self, key):
        return self.space[key]

    def __iter__(self):
        for key in self.spaces:
            yield key

    def __contains__(self, item):
        return self.contains(item)

    def __repr__(self):
        return "DummyObservationSpace()"

    def to_jsonable(self, sample_n):
        # serialize as dict-repr of vectors
        return self.spaces.to_dict()

    def from_jsonable(self, sample_n):
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret

    def __eq__(self, other):
        return isinstance(other, DummyObservationSpace) and self.spaces == other.spaces

# class DummyPreprocessor(Preprocessor):

#     @override(Preprocessor)
#     def _init_shape(self, obs_space, options):
#         return self._obs_space.shape

#     @override(Preprocessor)
#     def transform(self, observation):
#         self.check_shape(observation)
#         return observation

#     @override(Preprocessor)
#     def write(self, observation, array, offset):
#         array[offset:offset + self._size] = np.array(
#             observation, copy=False).ravel()

#     @property
#     @override(Preprocessor)
#     def observation_space(self):
#         return self._obs_space

