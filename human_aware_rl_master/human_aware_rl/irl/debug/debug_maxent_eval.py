import pickle
import os
from human_aware_rl.ppo.ppo_rllib_client import run

check_path='./result/human/T7/epoch=100.checkpoint'
os.path.isfile(check_path)

with open(check_path, 'rb') as file:
   checkpoint = pickle.load(file)
 
config = checkpoint['config']
reward_model = checkpoint["reward_model"]
irl_config = config['irl_params']
config['environment_params']['multi_agent_params']['custom_reward_func'] = reward_model.get_rewards

from human_aware_rl.rllib.rllib import OvercookedMultiAgent, reset_dummy_policy, save_trainer, gen_trainer_from_params

environment_params = config['environment_params']
model_params = config['model_params']
env = OvercookedMultiAgent.from_config(environment_params)

current_state = env.base_env.state
current_state.to_dict()

from overcooked_ai_py.mdp.actions import Action, Direction
joint_action = [Direction.SOUTH, Action.STAY]
joint_agent_action_info = [{}, {}]
next_state, mdp_infos = env.base_env.mdp.get_state_transition(current_state, joint_action)
env.base_env.state = next_state


