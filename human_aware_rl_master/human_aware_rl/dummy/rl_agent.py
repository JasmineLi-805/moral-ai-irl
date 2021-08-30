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
import gym


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

    def reset(self):
        """
        called after each iteration to reset agents to the initial state
        """
        if isinstance(self.model, MAIDumbAgentLeftCoop):
            self.model = MAIDumbAgentLeftCoop()
        elif isinstance(self.model, MAIDumbAgentRightCoop):
            self.model = MAIDumbAgentRightCoop()
        else:
            print(f'rl_agent.DummyPolicy.reset failed on type check')

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


def mai_dummy_feat_fn(state):
    featurized = {}
    pos = (5, 3)    # according to default value for MaiDummyAgent
    help_obj_name = 'onion'
    obj = state.objects.get(pos, None)

    player_pos = state.player_positions
    right_player_idx = 0
    if player_pos[1][0] > 5:
        right_player_idx = 1

    if state.players[right_player_idx].held_object:
        featurized['player_right_held_obj'] = 1
    else: 
        featurized['player_right_held_obj'] = 0

    if obj and obj.to_dict()['name'] == help_obj_name:  # 1, if the onion is at the help position 
        featurized['help_obj'] = 1
    else:
        featurized['help_obj'] = 0

    # print(featurized)
    return featurized

def get_mai_dummy_obs_space():
    return gym.spaces.Dict({"help_obj": gym.spaces.Discrete(2),
                            "player_right_held_obj":gym.spaces.Discrete(2)})
