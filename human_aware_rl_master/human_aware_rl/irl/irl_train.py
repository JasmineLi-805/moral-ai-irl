from math import e, inf
import sys, os

sys.path.append('/Users/jasmineli/Desktop/moral-ai-irl')
sys.path.append('/Users/jasmineli/Desktop/moral-ai-irl/human_aware_rl_master')
import pickle
import argparse
from human_aware_rl.irl.irl_agent import irlAppAgent
from human_aware_rl.ppo.ppo_rllib_client import run
from human_aware_rl_master.human_aware_rl.human.process_dataframes import *
from human_aware_rl_master.human_aware_rl.irl.reward_models import LinearReward
from human_aware_rl.dummy.rl_agent import *
from human_aware_rl.rllib.utils import get_base_ae
from overcooked_ai_py.agents.agent import AgentPair
from human_aware_rl.irl.config_model import get_train_config


def _apply_discount(states, gamma):
    result = states.copy()
    for i in range(len(states)):
        g = pow(gamma, len(states) - i - 1)
        result[i] = g * states[i]
    return result


def calculateFE(states, irl_config):
    result = np.sum(states, axis=0)
    return result

def _get_agent_featurized_states(states, joint_action, env):
    target_player_idx = 0
    num_game = len(states)
    # print(f'num games: {num_game}')
    all_feat = []
    for game, actions in zip(states,joint_action):
        feat_states = []
        for s,a in zip(game,actions):
            reward_features = env.irl_reward_state_encoding(s, a)
            feat_states.append(reward_features)
        feat_states = np.array(feat_states)
        feat_states = np.swapaxes(feat_states,0,1)
        all_feat.append(feat_states[target_player_idx])

    all_feat = np.array(all_feat)
    all_feat = np.sum(all_feat, axis=0)
    all_feat = all_feat / num_game
    return all_feat

def getMAIDummyFE(train_config, irl_config):
    mdp_params = train_config["environment_params"]["mdp_params"]
    env_params = train_config["environment_params"]["env_params"]
    ae = get_base_ae(mdp_params, env_params)
    env = ae.env

    states = []
    actions = []
    agents = [MAIToOnionShortAgent()]
    # agents = [MAIConditionedCoopLeftAgent()]
    for a in agents:
        agent_pair = AgentPair(a, MAIDummyRightCoopAgent())
        results = env.get_rollouts(agent_pair=agent_pair, num_games=1, display=True)
        states.append(results['ep_states'])
        actions.append(results['ep_actions'])

    # print(f'MAI actions traj num={len(actions)}, traj len={len(actions[0][0])}')
    act = []
    for traj in actions:
        temp = []
        for idx in traj[0]:
            temp.append([Action.ACTION_TO_INDEX[idx[0]], Action.ACTION_TO_INDEX[idx[1]]])
        act.append(temp)
    actions = act
    states = np.concatenate(states, axis=0)
    featurized_states = _get_agent_featurized_states(states,actions, env)
    feature_expectation = calculateFE(featurized_states, irl_config)
    print(f'expert FE={feature_expectation}')
    return feature_expectation

def getRLAgentFE(train_config, irl_config): #get the feature expectations of a new policy using RL agent
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
    results = None
    while True:
        try:
            results = run(train_config)
            break
        except Exception as e:
            print(e)

    rollout = results['evaluation']['states']
    actions = results['evaluation']['actions']
    # print(f'RL actions traj num={len(actions)}, traj len={len(actions[0])}')
    act = []
    for traj in actions:
        temp = []
        for idx in traj:
            temp.append([Action.ACTION_TO_INDEX[idx[0]], Action.ACTION_TO_INDEX[idx[1]]])
        act.append(temp)
    actions = act
    featurized_states = _get_agent_featurized_states(rollout, actions, env)
    feature_expectation = calculateFE(featurized_states, irl_config)
    return feature_expectation

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
    args = parse_args()
    assert args.trial
    trial = args.trial

    import tensorflow as tf
    print("tensorflow visible gpus:")
    print(tf.config.list_physical_devices('GPU'))

    cwd = os.getcwd()
    save_dir = f'{cwd}/result/T{trial}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # init 
    config = None
    n_epochs = args.epochs
    if not args.resume_from:
        accumulateT = []
        reward_obs_shape = 30         # change if reward shape changed.
        reward_model = LinearReward(reward_obs_shape)
        config = get_train_config(reward_func=reward_model.getRewards)
        irl_config = config['irl_params']
        expertFE = getMAIDummyFE(config, irl_config)    # the expert feature expectation (only uses mdp_params and env_params in config)
        irl_agent = irlAppAgent(expertFE=expertFE)
        i = 1
        bestT = inf
    else:
        checkpoint = load_checkpoint(args.resume_from)
        if "reward_obs_shape" in checkpoint:
            reward_obs_shape = checkpoint["reward_obs_shape"]
            reward_model = LinearReward(reward_obs_shape)
            reward_model.updateWeights(checkpoint["reward_model_weights"])
        config = checkpoint['config']
        config["environment_params"]["custom_reward_func"] = reward_model.getRewards
        irl_config = config['irl_params']
        irl_agent = checkpoint['irl_agent']
        i = checkpoint['curr_epoch'] + 1
        bestT = checkpoint['bestT']
        accumulateT = checkpoint['accumulateT']
    
    num_gpu = config['training_params']['num_gpus']
    print(f'num gpu = {num_gpu}')

    # randomly pick some policy, and compute the feature expectation
    agentFE = getRLAgentFE(config, irl_config)
    while i < n_epochs:
        print(f'----------------  {i}  ----------------')
        # compute t_i and W_i
        W, currentT = irl_agent.optimalWeightFinder(agentFE, reward_model.getRewards)
        accumulateT.append(currentT)
        if len(accumulateT) <= 20:
            print(f'the distances  :: {accumulateT}')
        else:
            print(f'the distances  :: {accumulateT[-20:]}')

        # if t_i <= epsilon, then terminate
        if currentT <= irl_config['epsilon']:
            final_pack = {
                # "reward_func": reward_model,
                "t": currentT,
                "bestT": currentT,
                "epsilon": irl_config['epsilon'],
                "config": config,
                "irl_agent": irl_agent,
                "curr_epoch": i,
                "accumulateT": accumulateT,
                "reward_model_weights": reward_model.weights,
                "reward_obs_shape": reward_obs_shape
            }
            file_name = 'final.pickle'
            with open(os.path.join(save_dir, file_name), 'wb') as save_file:
                pickle.dump(final_pack, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'final model saved to {os.path.join(save_dir, file_name)}')
            break
        
        # Using the RL algorithm, compute the optimal policy using reward weights W_i
        # and compute the feature expectation
        W = W.reshape((-1, 1))
        assert reward_model.weights.shape == W.shape
        reward_model.updateWeights(W)
        
        # config = get_train_config(reward_func=reward_model.getRewards)
        config["environment_params"]["custom_reward_func"] = reward_model.getRewards
        agentFE = getRLAgentFE(config, irl_config)
        
        # save file as pickle
        pack = {
            # "reward_func": reward_model,
            "currentT": currentT,
            "bestT": bestT,
            "epsilon": irl_config['epsilon'],
            "config": config,
            "irl_agent": irl_agent,
            "max_epoch": -1,
            "curr_epoch": i,
            "accumulateT": accumulateT,
            "reward_model_weights": reward_model.weights,
            "reward_obs_shape": reward_obs_shape
        }
        file_name = 'latest.pickle'
        with open(os.path.join(save_dir, file_name), 'wb') as save_file:
            pickle.dump(pack, save_file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'current model saved to {os.path.join(save_dir, file_name)}')
        
        if currentT < bestT:
            bestT = currentT
            pack["bestT"] = bestT
            file_name = 'best.pickle'
            with open(os.path.join(save_dir, file_name), 'wb') as save_file:
                pickle.dump(pack, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'best model saved to {os.path.join(save_dir, file_name)}')
            
        i += 1
    
    print('Final weights found, irl training completed.')

