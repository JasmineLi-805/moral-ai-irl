import os
import shutil

import pickle
import argparse
from human_aware_rl.ppo.ppo_rllib_client import run
from human_aware_rl_master.human_aware_rl.human.process_dataframes import *
from human_aware_rl.rllib.rllib import reset_dummy_policy, gen_trainer_from_params
from human_aware_rl_master.human_aware_rl.irl.reward_models import TorchLinearReward, TorchRNNReward, TorchLinCombReward
from human_aware_rl.dummy.rl_agent import *
from human_aware_rl.rllib.utils import get_base_ae
from human_aware_rl.irl.config_model import get_train_config
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        results = run_rl_training(train_config)
        states = results['evaluation']['states']
        actions = results['evaluation']['actions']
        scores = results['evaluation']['sparse_reward']
        actions = _convertAction2Index(actions)
        state_visit = getVisitation(states, actions, scores, env)
        return state_visit, results['evaluation']
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

    # make a copy of the config file
    # path = os.path.join(save_dir, f'config.py')
    # shutil.copy('config_model.py', path)

    # init 
    n_epochs = args.epochs

    if not args.resume_from:
        # make a copy of the config file
        path = os.path.join(save_dir, f'config.py')
        shutil.copy('config_model.py', path)

        print(f'initiating models and optimizers...')
        reward_obs_shape = torch.tensor([18])       # change if reward shape changed.
        # reward_model = TorchLinCombReward(reward_obs_shape)
        reward_model = TorchLinearReward(reward_obs_shape, n_h1=200)
        # reward_model = TorchRNNReward(n_input=reward_obs_shape, n_h1=200)
        optim = torch.optim.SGD(reward_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.999) 

        print(f'loading training configurations...')
        config = get_train_config()
        i = 0
        diff_log = {}

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
        scheduler = checkpoint['scheduler']
        
        print(f'loading configurations...')
        config = checkpoint['config']
        env = _loadEnvironment(config)
        i = checkpoint['current_epoch'] + 1 # advance to the next epoch
        if 'diff_log' in checkpoint:
            diff_log = checkpoint['diff_log']
        else:
            diff_log = {}
        
        print(f'getting expert trajectory and state visitation...')
        expert_state_visit = checkpoint['expert_svf']
        print(f'complete')

    # set the reward function used for RL training.
    config['environment_params']['multi_agent_params']['custom_reward_func'] = reward_model.get_rewards

    while i < n_epochs:
        print(f'iteration {i}, curr lr={scheduler.get_last_lr()}')
        # train a policy and get feature expectation
        agent_state_visit, eval_traj = getAgentVisitation(config, env)

        # compute the rewards and gradients for occurred states
        states, grad_r = getStatesAndGradient(expert_state_visit, agent_state_visit)
        reward = reward_model.forward(states)
        # reward, _hidden = reward_model.forward(states, None)
        assert reward.shape == grad_r.shape, f'reward={reward.shape}, grad_r={grad_r.shape}'
        
        # gradient descent
        optim.zero_grad()
        reward.backward(gradient=grad_r)
        optim.step()
        if scheduler:
            scheduler.step()

        d_mu = grad_r.detach().clone()
        d_mu = torch.square(d_mu)
        d_mu = torch.sum(d_mu)
        diff_log[i] = d_mu
        print(f'completed iteration {i}, diff: {d_mu}')

        i += 1
        if i % 5 == 0:
            checkpoint = {
                "reward_model": reward_model,
                "optimizer": optim,
                "config": config,
                "current_epoch": i,
                "expert_svf": expert_state_visit,
                "scheduler": scheduler,
                "diff_log": diff_log
            }
            file_name = f'epoch={i}.checkpoint'
            with open(os.path.join(save_dir, file_name), 'wb') as save_file:
                pickle.dump(checkpoint, save_file, protocol=pickle.HIGHEST_PROTOCOL)

            trajectory = [eval_traj]
            eval_file  = f'epoch={i}.trajectory'
            with open(os.path.join(save_dir, eval_file), 'wb') as save_file:
                pickle.dump(trajectory, save_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'training completed')
