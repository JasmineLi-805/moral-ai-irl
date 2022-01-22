import numpy as np

from human_aware_rl.irl import config
from human_aware_rl.irl.reward_models import LinearReward

from human_aware_rl.dummy.rl_agent import MAIDummyLeftCoopAgent, MAIDummyRightCoopAgent
from human_aware_rl.rllib.utils import get_base_ae
from overcooked_ai_py.agents.agent import AgentPair
from human_aware_rl.irl.irl_train import _apply_discount

from human_aware_rl.ppo import ppo_rllib_client

# TODO: Shouldn't this '#features' be set based on model input? LIKELY: 'len(feat_states[0][0])'
reward_model = LinearReward(num_in_feature=30)
all_config = config.get_train_config(reward_func=reward_model.getRewards)

# CREATE MAI_DUMMY_AGENTS AND GAME (in 'getMAIDummyFE' function)
mdp_params = all_config["environment_params"]["mdp_params"]
env_params = all_config["environment_params"]["env_params"]
agent_0_policy = MAIDummyLeftCoopAgent()
agent_1_policy = MAIDummyRightCoopAgent()
agent_pair = AgentPair(agent_0_policy, agent_1_policy)
agent_evaluator = get_base_ae(mdp_params, env_params)
game_environment = agent_evaluator.env

# GAME SETUP LOOKS GOOD!
game_environment.state.to_dict()

# RUN A GAME WITH THIS SETUP ALSO LOOKS GOOD!
results = game_environment.get_rollouts(agent_pair=agent_pair, num_games=1, display=False)
results['ep_infos'][0][-1]

# STATES' ENCODING (FEATURIZATION?)
states = results['ep_states'][0]
feat_states = []
for s in states:
    # using lossless feats
    reward_features = game_environment.irl_reward_state_encoding(s)
    feat_states.append(reward_features)

feat_states = np.array(feat_states)
feat_states = np.swapaxes(feat_states, 0, 1)

# CALCULATE FEATURE EXPECTATION: sum of featurized states
irl_config = all_config['irl_params']
gamma = irl_config['discount_factor']

player1 = _apply_discount(feat_states[0], gamma)
player2 = _apply_discount(feat_states[1], gamma)
# TODO: DOES THIS LOOK GOOD?
test_discount = _apply_discount(feat_states[0], 0.99)

np.sum(player1, axis=0)
np.sum(test_discount, axis=0)

