import sys, os
import shutil
import pickle
import argparse

from human_aware_rl_master.human_aware_rl.human.process_dataframes import *
from human_aware_rl.rllib.rllib import reset_dummy_policy, gen_trainer_from_params
from human_aware_rl_master.human_aware_rl.irl.reward_models import TorchLinearReward
from human_aware_rl.dummy.rl_agent import *
from human_aware_rl.rllib.utils import get_base_ae
from overcooked_ai_py.agents.agent import AgentPair
from human_aware_rl.irl.config_model import get_train_config

import torch

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

def run_rl_training(params):
    # Retrieve the tune.Trainable object that is used for the experiment
    trainer = gen_trainer_from_params(params)
    # Object to store training results in
    result = {}

    # Training loop
    for i in range(params['num_training_iters']):
        result = trainer.train()

        msg = result['episode_reward_mean']
        msg2 = result['episode_reward_max']
        msg3 = result['episode_reward_min']
        if i % 10 == 0:
            print(f'{i}: ep rew mean={msg}, max={msg2}, min={msg3}')
        trainer.workers.foreach_worker(lambda ev: reset_dummy_policy(ev.get_policy('dummy')))
    
    return result

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
    # agents = [MAICooperativeAgent()]
    agents = [MAINonCoopAgent()]

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
        results = run_rl_training(train_config)
        states = results['evaluation']['states']
        actions = results['evaluation']['actions']
        scores = results['evaluation']['sparse_reward']
        actions = _convertAction2Index(actions)
        state_visit = getVisitation(states, actions, env)
        return state_visit
    except Exception as e:
        print('ERROR: could not get Agent Visitation. --> ' + str(e))

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
    parser.add_argument('--checkpoint', type=str, default=None, help='pickle file to resume training')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print(f'Deep MaxEnt IRL evaluation starting...')
    print(f'can use gpu: {torch.cuda.is_available()}; device={device}')

    args = parse_args()
    
    print(f'loading model checkpoint from {args.checkpoint}...')
    checkpoint = load_checkpoint(args.checkpoint)
    
    print(f'retrieving reward model and optimizer...')
    reward_model = checkpoint["reward_model"]
    
    print(f'loading configurations...')
    config = checkpoint['config']
    env = _loadEnvironment(config)
    
    print(f'getting expert trajectory and state visitation...')
    expert_state_visit = checkpoint['expert_svf']
    print(f'complete')

    # set the reward function used for RL training.
    config['environment_params']['multi_agent_params']['custom_reward_func'] = reward_model.get_rewards
    config['evaluation_params']['display'] = True
    # config['num_training_iters'] = 150

    eplen = config['evaluation_params']['ep_length']
    print(f"config eval ep: {eplen}")

    print(f'start evaluating')
    # train a policy and get feature expectation
    agent_state_visit = getAgentVisitation(config, env)