import os, sys
sys.path.append(os.path.dirname('/Users/jasmineli/Desktop/moral-ai-irl/overcooked_demo_litw'))
from overcooked_demo_litw.server.game import MAIDumbAgent, MAIDumbAgentLeftCoop, MAIDumbAgentRightCoop
from overcooked_ai_py.mdp.actions import Action
from ray.rllib.policy import Policy as RllibPolicy
import numpy as np
from tensorflow.compat.v1.keras.backend import set_session, get_session



class DummyPolicy(RllibPolicy):
    """
    This wraps the preprogrammed Dummy Policy into an RllibPolicy
    """
    def __init__(self, observation_space, action_space, config):
        super(DummyPolicy, self).__init__(observation_space, action_space, config)

        possible_layout = ['mai_separate_coop_left', 'mai_separate_coop_right']
        assert config['layout'] in possible_layout
        self.model = MAIDumbAgentLeftCoop() if config['layout'] == 'mai_separate_coop_left' else MAIDumbAgentRightCoop()
        # self.context = self._create_execution_context()

        print('a DummyPolicy is instantiated')

    def _setup_shapes(self):
        # This is here to make the class compatible with both tuples or gym.Space objs for the spaces
        # Note: action_space = (len(Action.ALL_ACTIONS,)) is technically NOT the action space shape, which would be () since actions are scalars
        self.observation_shape = self.observation_space if type(self.observation_space) == tuple else self.observation_space.shape
        self.action_shape = self.action_space if type(self.action_space) == tuple else (self.action_space.n,)

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
            info (dict): if needed
        """
        next_action = self.model.action(obs_batch)
        return next_action[0], None, None

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

