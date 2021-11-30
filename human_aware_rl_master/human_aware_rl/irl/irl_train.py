from math import inf
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
from human_aware_rl.irl.config import get_train_config

def calculateFE(states, irl_config):
    def _apply_discount(states, gamma):
        result = states.copy()
        for i in range(len(states)):
            g = pow(gamma, len(states) - i - 1)
            result[i] = g * states[i]
        return states
    
    gamma = irl_config['discount_factor']
    result = _apply_discount(states, gamma)
    result = np.sum(result, axis=0)
    return result

def getMAIDummyFE(train_config, irl_config):
    mdp_params = train_config["environment_params"]["mdp_params"]
    env_params = train_config["environment_params"]["env_params"]
    agent_0_policy = MAIDummyLeftCoopAgent()
    agent_1_policy = MAIDummyRightCoopAgent()
    agent_pair = AgentPair(agent_0_policy, agent_1_policy)

    ae = get_base_ae(mdp_params, env_params)
    env = ae.env
    results = env.get_rollouts(agent_pair=agent_pair, num_games=1, display=False)
    
    states = results['ep_states'][0]
    # check which index corresponds to agent on the left and vice versa
    left_idx = -1 
    right_idx = -1
    pos = states[0].player_positions
    if pos[0] == (3,1) and pos[1] == (8,1):
        right_idx = 1
        left_idx = 0
    elif pos[1] == (3,1) and pos[0] == (8,1):
        right_idx = 0
        left_idx = 1
    assert left_idx != -1 and right_idx != -1
    # print(f'RL agent position={pos}, left idx={left_idx}, right idx={right_idx}')

    feat_states = []
    for s in states:
        feat = env.featurize_state_mdp(s)
        feat_states.append(feat)
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

def getRLAgentFE(train_config, irl_config): #get the feature expectations of a new policy using RL agent
    '''
    Trains an RL agent with the current reward function. 
    Then rolls out one trial of the trained agent and calculate the feature expectation of the RL agent.
    - train_config: the configuration taken by the rllib trainer
    
    Returns the feature expectation.
    '''
    layout = irl_config['layout']
    # train and get rollouts
    results = run(train_config)
    agent_rollout = results['evaluation']['states']

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
        if state == agent_rollout[0]:
            print(layout)
            print(state.player_positions)
        res = env.featurize_state_mdp(state)
        feat_states.append(res)
    feat_states = np.array(feat_states)
    feat_states = np.swapaxes(feat_states,0,1)
    
    if layout == 'mai_separate_coop_right':
        right_state = calculateFE(feat_states[right_idx], irl_config)
        # print(f'RL right FE shape = {right_state.shape}')
        return right_state
    else:
        assert layout == 'mai_separate_coop_left'
        left_state = calculateFE(feat_states[left_idx], irl_config)
        # print(f'RL left FE shape = {left_state.shape}')
        return left_state

def load_checkpoint(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'rb') as file:
        checkpoint = pickle.load(file)
    return checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--resume_from', type=str, default=None, help='pickle file to resume training')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # inputs, targets, seq_lens = load_data()
    # print(f'input = {inputs.shape}')
    # print(f'targets = {targets.shape}')
    TRIAL = 10

    cwd = os.getcwd()
    save_dir = f'{cwd}/result/T{TRIAL}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # init 
    config = None
    args = parse_args()
    if not args.resume_from:
        reward_model = LinearReward(96)
        config = get_train_config(reward_func=reward_model.getRewards)
        irl_config = config['irl_params']
        expertFE = getMAIDummyFE(config, irl_config)    # the expert feature expectation (only uses mdp_params and env_params in config)
        irl_agent = irlAppAgent(expertFE=expertFE)
        i = 0
        bestT = inf
    else:
        checkpoint = load_checkpoint(args.resume_from)
        reward_model = checkpoint['reward_func']
        config = checkpoint['config']
        irl_config = config['irl_params']
        irl_agent = checkpoint['irl_agent']
        i = checkpoint['curr_epoch']
        bestT = checkpoint['bestT']

    while True:
        print(f'----------------  {i}  ----------------')
        agentFE = getRLAgentFE(config, irl_config)
        # print(agentFE)
        W, currentT = irl_agent.optimalWeightFinder(agentFE, reward_model.getRewards)
        
        W = W.reshape((-1, 1))
        assert reward_model.weights.shape == W.shape
        reward_model.updateWeights(W)

        if currentT <= irl_config['epsilon']:
            break
        
        # save file as pickle
        pack = {
            "reward_func": reward_model,
            "currentT": currentT,
            "bestT": bestT,
            "epsilon": irl_config['epsilon'],
            "config": config,
            "irl_agent": irl_agent,
            "max_epoch": -1,
            "curr_epoch": i
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



# DEFAULT_DATA_PARAMS = {
#     "layouts": ["cramped_room"],
#     "check_trajectories": False,
#     "featurize_states": True,
#     "data_path": CLEAN_2019_HUMAN_DATA_TRAIN
# }


# def load_data():
#     def _pad(sequences, maxlen=None, default=0):
#         if not maxlen:
#             maxlen = max([len(seq) for seq in sequences])
#         for seq in sequences:
#             pad_len = maxlen - len(seq)
#             seq.extend([default]*pad_len)
#         return sequences

#     processed_trajs = get_human_human_trajectories(
#         **DEFAULT_DATA_PARAMS, silent=False)
#     print(len(processed_trajs))
#     for k in processed_trajs:
#         print(k)
#     inputs, targets = processed_trajs["ep_states"], processed_trajs["ep_actions"]
#     print(f'inputs = {type(inputs)}; len = {len(inputs)}')
#     print(f'targets = {type(targets)}; len = {len(targets)}')

#     # sequence matters, pad the shorter ones with init state
#     seq_lens = np.array([len(seq) for seq in inputs])
#     seq_padded = _pad(inputs, default=np.zeros((len(inputs[0][0],))))
#     targets_padded = _pad(targets, default=np.zeros(1))
#     seq_t = np.dstack(seq_padded).transpose((2, 0, 1))
#     targets_t = np.dstack(targets_padded).transpose((2, 0, 1))

#     return seq_t, targets_t, seq_lens
