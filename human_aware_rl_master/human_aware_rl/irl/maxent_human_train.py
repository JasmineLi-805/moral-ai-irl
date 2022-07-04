from cmath import exp
from math import e, inf
import glob
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

def getVisitation(states, joint_action, env):
    target_player_idx = 0
    num_game = len(states)
    freq = {}
    for game, actions in zip(states,joint_action):
        for s,a in zip(game,actions):
            reward_features = env.irl_reward_state_encoding(s, a)[target_player_idx]
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
    agents = [MAIToOnionLongAgent()]
    for a in agents:
        agent_pair = AgentPair(a, MAIDummyRightCoopAgent())
        results = env.get_rollouts(agent_pair=agent_pair, num_games=1, display=False)
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

def getAgentVisitation(train_config, irl_config): #get the feature expectations of a new policy using RL agent
    '''
    Trains an RL agent with the current reward function. 
    Then rolls out one trial of the trained agent and calculate the feature expectation of the RL agent.
    - train_config: the configuration taken by the rllib trainer
    
    Returns the feature expectation.
    '''
    mdp_params = train_config["environment_params"]["mdp_params"]
    env_params = train_config["environment_params"]["env_params"]
    ae = get_base_ae(mdp_params, env_params)
    env = ae.env
    # train and get rollouts
    try:
        results = run(train_config)
    except Exception as e:
        print(e)

    states = results['evaluation']['states']
    actions = results['evaluation']['actions']
    # print(f'RL actions traj num={len(actions)}, traj len={len(actions[0])}')
    act = []
    for traj in actions:
        temp = []
        for idx in traj:
            temp.append([Action.ACTION_TO_INDEX[idx[0]], Action.ACTION_TO_INDEX[idx[1]]])
        act.append(temp)
    actions = act
    state_visit = getVisitation(states, actions, env)
    return state_visit

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
        states.append(state)
        grad.append(visit[s])
    states = torch.stack(states)
    grad = torch.tensor(grad, dtype=torch.float)
    grad = torch.unsqueeze(grad, dim=1)

    return states, grad

def viewReward(reward_model):
    input = torch.zeros(30, 30)
    for i in range(30):
        input[i][i] = 1
    rewards = reward_model.get_rewards(input)
    
    rewards = torch.reshape(rewards, (6,5))
    rewards = rewards.transpose(0,1)
    print(rewards)
    
    plt.imshow(rewards, cmap='hot', interpolation='nearest')
    plt.savefig("reward.png")


def load_checkpoint(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'rb') as file:
        checkpoint = pickle.load(file)
    return checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-t', '--trial', type=int, help='trial num')
    parser.add_argument('--resume_from', type=str, default=None, help='pickle file to resume training')
    parser.add_argument('--epochs', type=int, default=100, help='total number of epochs to train')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print(f'Deep MaxEnt IRL training starting...')
    print(f'can use gpu: {torch.cuda.is_available()}; device={device}')

    args = parse_args()
    # assert args.trial
    # trial = args.trial

    # directory to save results
    # cwd = os.getcwd()
    # save_dir = f'{cwd}/result/T{trial}'
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)

    # init 
    n_epochs = args.epochs
    
    print(f'initiating models and optimizers...')
    reward_obs_shape = torch.tensor([30])       # change if reward shape changed.
    reward_model = TorchLinearReward(reward_obs_shape)
    # reward_model.to(device)
    optim = torch.optim.SGD(reward_model.parameters(), lr=0.02, momentum=0.9, weight_decay=0.9)
    print(f'complete')
    
    config = get_train_config(reward_func=reward_model.get_rewards)
    irl_config = config['irl_params']
    
    print(f'getting expert trajectory and state visitation...')
    expert_state_visit = getExpertVisitation(config, irl_config)    # only uses mdp_params and env_params in config
    print(f'complete')
    for i in range(n_epochs):
        if i % 10 == 0:
            print(f'iteration {i}')
        # train a policy and get feature expectation
        config["environment_params"]["custom_reward_func"] = reward_model.get_rewards
        agent_state_visit = getAgentVisitation(config, irl_config)

        # compute the rewards and gradients for occurred states
        states, grad_r = getStatesAndGradient(expert_state_visit, agent_state_visit)
        reward = reward_model.forward(states)
        
        # gradient descent
        optim.zero_grad()
        reward.backward(gradient=grad_r)
        optim.step()
    print(f'training completed')
    
    viewReward(reward_model)
