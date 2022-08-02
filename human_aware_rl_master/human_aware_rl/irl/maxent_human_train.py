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
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _apply_discount(states, gamma):
    result = states.copy()
    for i in range(len(states)):
        g = pow(gamma, len(states) - i - 1)
        result[i] = g * states[i]
    return result

def _loadEnvironment(config):
    mdp_params = config["environment_params"]["mdp_params"]
    env_params = config["environment_params"]["env_params"]
    ae = get_base_ae(mdp_params, env_params)
    env = ae.env
    
    return env

def _loadProcessedHumanData(data_path, view_traj=False):
    assert os.path.isfile(data_path)
    with open(data_path, 'rb') as file:
        human_data = pickle.load(file)
    
    gridworld = human_data['gridworld']
    trajectory = human_data['trajectory']

    states = []
    actions = []
    scores = []
    for i in range(len(trajectory)):
        state = []
        action = []
        score = []
        for j in range(len(trajectory[i])):
            state_dict = trajectory[i][j]
            s = state_dict['state']
            a = state_dict['joint_action']
            sc = state_dict['score']

            s = OvercookedState.from_dict(s)
            state.append(s)
            action.append(a)
            score.append(sc)
            
            if view_traj:
                print(gridworld.state_string(s))
        states.append(state)
        actions.append(action)
        scores.append(score)

    assert len(states) == len(trajectory)
    assert len(actions) == len(trajectory)
    assert len(scores) == len(trajectory)
    return states, actions, scores

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

def getVisitation(states, joint_action, scores, env):
    target_player_idx = 0
    num_game = len(states)
    freq = {}
    for game, actions, score in zip(states,joint_action, scores):
        for s,a,sc in zip(game,actions, score):
            reward_features = env.human_coop_state_encoding(s, a, sc)[target_player_idx]
            reward_features = tuple(reward_features)
            if reward_features not in freq:
                freq[reward_features] = 0
            freq[reward_features] += 1
    
    for state in freq:
        freq[state] /= num_game
    return freq

def getExpertVisitation(env, data_path):
    states, actions, scores = _loadProcessedHumanData(data_path, view_traj=False)
    actions = _convertAction2Index(actions)
    state_visit = getVisitation(states,actions, scores, env)
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
        state_visit = getVisitation(states, actions, scores, env)
        return state_visit
    except Exception as e:
        print('ERROR: could not get Agent Visitation. -->' + e.with_traceback())

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
    save_dir = f'{cwd}/result/human/T{trial}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # init 
    n_epochs = args.epochs

    if not args.resume_from:
        print(f'initiating models and optimizers...')
        reward_obs_shape = torch.tensor([15])       # change if reward shape changed.
        reward_model = TorchLinearReward(reward_obs_shape)
        optim = torch.optim.SGD(reward_model.parameters(), lr=0.02, momentum=0.9, weight_decay=0.9)

        print(f'loading training configurations...')
        config = get_train_config()

        print(f'getting expert trajectory and state visitation...')
        env = _loadEnvironment(config)
        expert_state_visit = getExpertVisitation(env, data_path)
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

    for i in range(n_epochs):
        if i % 10 == 0:
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