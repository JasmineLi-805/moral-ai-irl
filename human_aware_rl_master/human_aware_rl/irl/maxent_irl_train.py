from cmath import exp
from math import e, inf
import sys, os
import shutil

sys.path.append('/Users/jasmineli/Desktop/moral-ai-irl')
sys.path.append('/Users/jasmineli/Desktop/moral-ai-irl/human_aware_rl_master')
import pickle
import argparse
import matplotlib.pyplot as plt
from human_aware_rl.ppo.ppo_rllib_client import run
from human_aware_rl_master.human_aware_rl.human.process_dataframes import *
from human_aware_rl_master.human_aware_rl.irl.reward_models import TorchLinearReward
from human_aware_rl.dummy.rl_agent import *
from human_aware_rl.rllib.utils import get_base_ae
from overcooked_ai_py.agents.agent import AgentPair
from human_aware_rl.irl.config_model import get_train_config

import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _loadEnvironment(config):
    mdp_params = config["environment_params"]["mdp_params"]
    env_params = config["environment_params"]["env_params"]
    ae = get_base_ae(mdp_params, env_params)
    env = ae.env
    
    return env

def _convertAction2Index(actions):
    act = []
    for traj in actions:
        temp = []
        for idx in traj:
            act_0 = tuple(idx[0]) if type(idx[0]) == list else idx[0]
            act_1 = tuple(idx[1]) if type(idx[1]) == list else idx[1]
            temp.append([Action.ACTION_TO_INDEX[act_0], Action.ACTION_TO_INDEX[act_1]])
        act.append(temp)
    return act

def getVisitation(states, joint_action, env):
    target_player_idx = 0
    num_game = len(states)
    freq = {}
    for game, actions in zip(states,joint_action):
        for s,a in zip(game,actions):
            reward_features = env.human_coop_state_encoding(s, a, 0)[target_player_idx]
            reward_features = tuple(reward_features)
            if reward_features not in freq:
                freq[reward_features] = 0
            freq[reward_features] += 1
    
    for state in freq:
        freq[state] /= num_game
    return freq

def getExpertVisitation(train_config, irl_config):
    mdp_params = train_config["environment_params"]["mdp_params"]
    env_params = train_config["environment_params"]["env_params"]
    ae = get_base_ae(mdp_params, env_params)
    env = ae.env

    states = []
    actions = []
    # agents = [ MAIToOnionLongAgent(), MAIToOnionShortAgent]
    agents = [MAICooperativeAgent()]

    for a in agents:
        agent_pair = AgentPair(a, MAIDummyRightCoopAgent())
        results = env.get_rollouts(agent_pair=agent_pair, num_games=1, display=True)
        states.append(results['ep_states'])
        actions.append(results['ep_actions'])

    act = []
    for traj in actions:
        temp = []
        for idx in traj[0]:
            temp.append([Action.ACTION_TO_INDEX[idx[0]], Action.ACTION_TO_INDEX[idx[1]]])
        act.append(temp)
    actions = act
    states = np.concatenate(states, axis=0)
    state_visit = getVisitation(states,actions, env)
    return state_visit

def getAgentVisitation(train_config, env): #get the feature expectations of a new policy using RL agent
    '''
    Trains an RL agent with the current reward function. 
    Then rolls out one trial of the trained agent and calculate the feature expectation of the RL agent.
    - train_config: the configuration taken by the rllib trainer
    
    Returns the feature expectation.
    '''
    # train and get rollouts
    try:
        results = run(train_config)
        states = results['evaluation']['states']
        actions = results['evaluation']['actions']
        scores = results['evaluation']['sparse_reward']
        actions = _convertAction2Index(actions)
        state_visit = getVisitation(states, actions, env)
        return state_visit
    except Exception as e:
        print('ERROR: could not get Agent Visitation. -->' + str(e))

def getStatesAndGradient(expert_sv, agent_sv):
    # calculate the gradient for each of the state: (mu_agent - mu_expert)
    visit = {}
    for state in agent_sv:
        visit[state] = agent_sv[state]
    for state in expert_sv:
        if state not in visit:
            visit[state] = 0.0
        visit[state] -= expert_sv[state]
    
    # organize into NN input
    states = []
    grad = []
    for s in visit:
        state = torch.tensor(s, dtype=torch.float)
        # state.to(device)
        states.append(state)
        grad.append(visit[s])
    states = torch.stack(states)
    grad = torch.tensor(grad, dtype=torch.float)
    grad = torch.unsqueeze(grad, dim=1)

    return states, grad

def load_checkpoint(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'rb') as file:
        checkpoint = pickle.load(file)
    return checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-t', '--trial', type=str, help='trial index')
    parser.add_argument('--data', type=str, help='path to the data')
    parser.add_argument('--resume_from', type=str, default=None, help='pickle file to resume training')
    parser.add_argument('--epochs', type=int, default=100, help='total number of epochs to train')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print(f'Deep MaxEnt IRL training starting...')
    print(f'can use gpu: {torch.cuda.is_available()}; device={device}')

    args = parse_args()
    assert args.trial
    trial = args.trial
    data_path = args.data
    
    # directory to save results
    cwd = os.getcwd()
    save_dir = f'{cwd}/result/bot/T{trial}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # init 
    n_epochs = args.epochs

    if not args.resume_from:
        print(f'initiating models and optimizers...')
        reward_obs_shape = torch.tensor([17])       # change if reward shape changed.
        reward_model = TorchLinearReward(reward_obs_shape)
        optim = torch.optim.SGD(reward_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.9)

        print(f'loading training configurations...')
        config = get_train_config()
        i = 0

        print(f'getting expert trajectory and state visitation...')
        env = _loadEnvironment(config)
        expert_state_visit = getExpertVisitation(config, config["irl_params"])
        print(f'complete')
    else:
        print(f'loading model checkpoint from {args.resume_from}...')
        checkpoint = load_checkpoint(args.resume_from)
        
        print(f'retrieving reward model and optimizer...')
        reward_model = checkpoint["reward_model"]
        optim = checkpoint['optimizer']
        
        print(f'loading configurations...')
        config = checkpoint['config']
        env = _loadEnvironment(config)
        i = checkpoint['current_epoch'] + 1 # advance to the next epoch
        
        print(f'getting expert trajectory and state visitation...')
        expert_state_visit = checkpoint['expert_svf']
        print(f'complete')

    # make a copy of the config file
    path = os.path.join(save_dir, f'config.py')
    shutil.copy('config_model.py', path)

    # set the reward function used for RL training.
    config['environment_params']['multi_agent_params']['custom_reward_func'] = reward_model.get_rewards

    while i < n_epochs:
        print(f'iteration {i}')
        # train a policy and get feature expectation
        agent_state_visit = getAgentVisitation(config, env)

        # compute the rewards and gradients for occurred states
        states, grad_r = getStatesAndGradient(expert_state_visit, agent_state_visit)
        reward = reward_model.forward(states)
        
        # gradient descent
        optim.zero_grad()
        reward.backward(gradient=grad_r)
        optim.step()

        i += 1
        if i % 5 == 0:
            checkpoint = {
                "reward_model": reward_model,
                "optimizer": optim,
                "config": config,
                "current_epoch": i,
                "expert_svf": expert_state_visit,
            }
            file_name = f'epoch={i}.checkpoint'
            with open(os.path.join(save_dir, file_name), 'wb') as save_file:
                pickle.dump(checkpoint, save_file, protocol=pickle.HIGHEST_PROTOCOL)

    final = {
        "reward_model": reward_model,
        "optimizer": optim,
        "config": config,
        "current_epoch": n_epochs,
        "expert_svf": expert_state_visit,
    }
    file_name = f'final.checkpoint'
    with open(os.path.join(save_dir, file_name), 'wb') as save_file:
        pickle.dump(final, save_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'training completed')
