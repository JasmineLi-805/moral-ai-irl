import os, sys
from typing import Dict

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from ray.rllib.models.preprocessors import Preprocessor
sys.path.append(os.path.dirname('/Users/jasmineli/Desktop/moral-ai-irl/overcooked_demo_litw'))
sys.path.append(os.path.dirname('/home/jasmine/moral-ai-irl/overcooked_demo_litw'))
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
        layout = config['layout']
        # the 'left' and 'right' in the layout name refers to the human player's position
        possible_layout = ['mai_separate_coop_left', 'mai_separate_coop_right']
        assert config['layout'] in possible_layout
        if config['layout']== 'mai_separate_coop_right':
            print(f'DummyPolicy: layout={layout}, agent=MAIDumbAgentLeftCoop')
            self.model = MAIDumbAgentLeftCoop() 
        elif config['layout']== 'mai_separate_coop_left':
            print(f'DummyPolicy: layout={layout}, agent=MAIDumbAgentRightCoop') 
            self.model = MAIDumbAgentRightCoop()
        # self.context = self._create_execution_context()

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

def mai_dummy_feat_fn(state):
    featurized = {}
    pos = (5, 3)    # according to default value for MaiDummyAgent
    help_obj_name = 'onion'
    obj = state.objects.get(pos, None)

    if state.players[1].held_object:
        featurized['player_1_held_obj'] = 1
    else: 
        featurized['player_1_held_obj'] = 0

    if obj and obj.to_dict()['name'] == help_obj_name:
        featurized['help_obj'] = 1
    else:
        featurized['help_obj'] = 0
    return featurized

