import os
import argparse
import pickle
import matplotlib.pyplot as plt

from human_aware_rl.ppo.ppo_rllib_client import run
from human_aware_rl_master.human_aware_rl.human.process_dataframes import *
from human_aware_rl_master.human_aware_rl.irl.reward_models import LinearReward
from human_aware_rl.dummy.rl_agent import *
from human_aware_rl.rllib.utils import get_base_ae
from overcooked_ai_py.agents.agent import AgentPair
from human_aware_rl.irl.config_model import get_train_config
from human_aware_rl.irl.irl_train import calculateFE

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to pickle file')
    args = parser.parse_args()
    return args

def load_checkpoint(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'rb') as file:
        checkpoint = pickle.load(file)
    return checkpoint

def run_rl_training(train_config):
    # train and get rollouts
    results = None
    while True:
        try:
            results = run(train_config)
            break
        except Exception as e:
            print(e)

    game_state = results['evaluation']['states']
    return game_state

def featurize_game_state(agent_rollout, train_config, irl_config):
    left_idx = -1
    right_idx = -1
    pos = agent_rollout[0].player_positions
    if pos[0] == (3,1) and pos[1] == (8,1):
        right_idx = 1
        left_idx = 0
    elif pos[1] == (3,1) and pos[0] == (8,1):
        right_idx = 0
        left_idx = 1
    assert left_idx != -1 and right_idx != -1
    # print(f'RL agent position={pos}, left idx={left_idx}, right idx={right_idx}')

    # featurize states
    mdp_params = train_config["environment_params"]["mdp_params"]
    env_params = train_config["environment_params"]["env_params"]
    ae = get_base_ae(mdp_params, env_params)
    env = ae.env
    feat_states = []
    for state in agent_rollout:
        # using lossless feats
        reward_features = env.irl_reward_state_encoding(state)
        feat_states.append(reward_features)

    feat_states = np.array(feat_states)
    feat_states = np.swapaxes(feat_states,0,1)
    
    layout = irl_config['layout']
    if layout == 'mai_separate_coop_right':
        right_state = calculateFE(feat_states[right_idx], irl_config)
        # print(f'RL right FE shape = {right_state.shape}')
        return right_state
    else:
        assert layout == 'mai_separate_coop_left'
        left_state = calculateFE(feat_states[left_idx], irl_config)
        # print(f'RL left FE shape = {left_state.shape}')
        return left_state

def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    # checkpoint = load_checkpoint('/Users/jasmineli/Desktop/moral-ai-irl/result/T0/latest.pickle')
    
    # run the training with 
    if "reward_obs_shape" in checkpoint:
        reward_obs_shape = checkpoint["reward_obs_shape"]
        reward_model = LinearReward(reward_obs_shape)
        reward_model.updateWeights(checkpoint["reward_model_weights"])
    else:
        reward_model = checkpoint["reward_func"]
    config = get_train_config(reward_func=reward_model.getRewards)
    config['training_params']['num_gpus']=0
    # config['num_training_iters'] = 10
    # config['evaluation_interval'] = 10
    run_rl_training(config)

    # display T values
    if 'accumulateT' in checkpoint:
        plt.plot(checkpoint['accumulateT'])
        plt.title('epoch vs. max margin')
        plt.ylabel('max margin')
        plt.xlabel('epoch')
        plt.show()

if __name__ == "__main__":
    main()
    

    

