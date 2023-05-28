import os, sys
from typing import Dict
from overcooked_ai_py.agents.agent import Agent

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, SoupState
from ray.rllib.models.preprocessors import Preprocessor
sys.path.append(os.path.dirname('/Users/jasmineli/Desktop/moral-ai-irl/overcooked_demo_litw'))
sys.path.append(os.path.dirname('/homes/iws/jl9985/moral-ai-irl/overcooked_demo_litw'))
sys.path.append(os.path.dirname('/home/jasmine/moral-ai-irl/overcooked_demo_litw'))
from overcooked_demo_litw.server.game import *
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
        possible_layout = ['mai_separate_coop_left', 'mai_separate_coop_right', 'coop_experiment_1', 'vertical_kitchen']
        assert config['layout'] in possible_layout
        if config['layout']== 'mai_separate_coop_right':
            print(f'DummyPolicy: layout={layout}, agent=MAIDumbAgentLeftCoop')
            self.model = MAIDumbAgentLeftCoop()
        elif config['layout'] == 'vertical_kitchen':
            print(f'DummyPolicy: layout={layout}, agent=VerticalRightCoop') 
            self.model = VerticalRightCoop()
        else:
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
        elif isinstance(self.model, VerticalRightCoop):
            self.model = VerticalRightCoop()
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
    
    # 'help_obj': 1 if the onion is at the help position
    pos = (5, 3)
    help_obj_name = 'onion'
    obj = state.objects.get(pos, None)
    if obj and obj.to_dict()['name'] == help_obj_name:
        featurized['help_obj'] = 1
    else:
        featurized['help_obj'] = 0

    # 'player_right_held_obj': 1 if the right agent is holding something
    player_pos = state.player_positions
    right_player_idx = 0
    if player_pos[1][0] > 5:
        right_player_idx = 1
    if state.players[right_player_idx].held_object:
        featurized['player_right_held_obj'] = 1
    else: 
        featurized['player_right_held_obj'] = 0

    left_stove_pos = (3, 0)
    left_stove = state.objects.get(left_stove_pos, None)
    if left_stove and isinstance(left_stove, SoupState) and left_stove.is_ready:
        featurized['soup_ready_left'] = 1
    else:
        featurized['soup_ready_left'] = 0
    
    right_stove_pos = (8, 0)
    right_stove = state.objects.get(right_stove_pos, None)
    if right_stove and isinstance(right_stove, SoupState) and right_stove.is_ready:
        featurized['soup_ready_right'] = 1
    else:
        featurized['soup_ready_right'] = 0

    # print(featurized)
    return featurized

def get_mai_dummy_obs_space():
    return gym.spaces.Dict({"help_obj": gym.spaces.Discrete(2),
                            "player_right_held_obj":gym.spaces.Discrete(2),
                            "soup_ready_left": gym.spaces.Discrete(2),
                            "soup_ready_right": gym.spaces.Discrete(2)})

class MAIDummyLeftCoopAgent(Agent):
    
    def __init__(self):
        self.agent = MAIDumbAgentLeftCoop()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None

class MAIDummyRightCoopAgent(Agent):
    
    def __init__(self):
        self.agent = MAIDumbAgentRightCoop()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None


##################
## IRL Subtasks ##
##################

# 1.1 clockwise walk-only agent
class MAIClockwiseLeftAgent(Agent):
    
    def __init__(self):
        self.agent = MAIClockwiseLeft()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None

# 1.2
class MAIToOnionShortAgent(Agent):
    
    def __init__(self):
        self.agent = ToOnionShort()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None

class MAIToOnionLongAgent(Agent):
    
    def __init__(self):
        self.agent = ToOnionLong()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None

######
# layout: coop_experiment_1

class MAICoopLeftAgent(Agent):
    
    def __init__(self):
        self.agent = CoopSendOnion()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None

class MAINonCoopAgent(Agent):
    
    def __init__(self):
        self.agent = NonCooperativeAgent()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None

class MAICooperativeAgent(Agent):
    
    def __init__(self):
        self.agent = CooperativeAgent()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None

class MAIEvaluationAgent(Agent):
    
    def __init__(self):
        self.agent = MAIEvalAgent()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None

class MAIStayAgent(Agent):
    
    def __init__(self):
        self.agent = StayAI()

    def action(self, state):
        act = self.agent.action(state)
        return act[0], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None