import sys
sys.path.append('/Users/jasmineli/Desktop/moral-ai-irl')
sys.path.append('/Users/jasmineli/Desktop/moral-ai-irl/human_aware_rl_master')
from human_aware_rl.irl.irl_agent import irlAppAgent
from human_aware_rl.ppo.ppo_rllib import RllibLSTMPPOModel, RllibPPOModel
from human_aware_rl.ppo.ppo_rllib_client import run
from ray.tune.result import CONFIG_PREFIX
from human_aware_rl_master.human_aware_rl.human.process_dataframes import *
from human_aware_rl_master.human_aware_rl.irl.reward_models import LinearReward
from human_aware_rl.dummy.rl_agent import *
from human_aware_rl.rllib.utils import get_base_ae
from human_aware_rl.rllib.rllib import gen_trainer_from_params
from overcooked_ai_py.agents.agent import AgentPair

LOCAL_TESTING = False
GERLACH = True

def _env_creator(env_config):
    # Re-import required here to work with serialization
    from human_aware_rl.rllib.rllib import OvercookedMultiAgent 
    return OvercookedMultiAgent.from_config(env_config)

def get_train_config(reward_func):
    ### Model params ###

    # Whether dense reward should come from potential function or not
    use_phi = True

    # whether to use recurrence in ppo model
    use_lstm = False

    # Base model params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    # LSTM memory cell size (only used if use_lstm=True)
    CELL_SIZE = 256

    # whether to use D2RL https://arxiv.org/pdf/2010.09163.pdf (concatenation the result of last conv layer to each hidden layer); works only when use_lstm is False
    D2RL = False
    ### Training Params ###

    num_workers = 12 if not LOCAL_TESTING else 2

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0]

    # Placeholder for random for current trial
    seed = None

    # Number of gpus the central driver should use
    num_gpus = 0 if LOCAL_TESTING else 1

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    train_batch_size = 4800 if not LOCAL_TESTING else 800

    # size of minibatches we divide up each batch into before
    # performing gradient steps
    sgd_minibatch_size = 1200 if not LOCAL_TESTING else 800

    # Rollout length
    rollout_fragment_length = 400
    
    # Whether all PPO agents should share the same policy network
    shared_policy = True

    # Number of training iterations to run
    num_training_iters = 200 if not LOCAL_TESTING else 10

    # Stepsize of SGD.
    lr = 5e-5

    # Learning rate schedule.
    lr_schedule = None

    # If specified, clip the global norm of gradients by this amount
    grad_clip = 0.1

    # Discount factor
    gamma = 0.99

    # Exponential decay factor for GAE (how much weight to put on monte carlo samples)
    # Reference: https://arxiv.org/pdf/1506.02438.pdf
    lmbda = 0.98

    # Whether the value function shares layers with the policy model
    vf_share_layers = True

    # How much the loss of the value network is weighted in overall loss
    vf_loss_coeff = 1e-4

    # Entropy bonus coefficient, will anneal linearly from _start to _end over _horizon steps
    entropy_coeff_start = 0.2
    entropy_coeff_end = 0.1
    entropy_coeff_horizon = 3e5

    # Initial coefficient for KL divergence.
    kl_coeff = 0.2

    # PPO clipping factor
    clip_param = 0.05

    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    num_sgd_iter = 8 if not LOCAL_TESTING else 1

    # How many trainind iterations (calls to trainer.train()) to run before saving model checkpoint
    save_freq = -1  # do not store intermediate RL agent results

    # How many training iterations to run between each evaluation
    evaluation_interval = 200 if not LOCAL_TESTING else 10

    # How many timesteps should be in an evaluation episode
    evaluation_ep_length = 400

    # Number of games to simulation each evaluation
    evaluation_num_games = 1

    # Whether to display rollouts in evaluation
    evaluation_display = True

    # Where to store model checkpoints and training stats
    results_dir = "/Users/jasmineli/Desktop/moral-ai-irl/result"
    if GERLACH:
        results_dir = "/home/jasmine/moral-ai-irl/result"
    

    # Whether tensorflow should execute eagerly or not
    eager = False

    # Whether to log training progress and debugging info
    verbose = False

    ### Environment Params ###
    # Which overcooked level to use
    layout_name = "mai_separate_coop_right"

    # all_layout_names = '_'.join(layout_names)

    # Name of directory to store training results in (stored in ~/ray_results/<experiment_name>)

    params_str = str(use_phi) + "_nw=%d_vf=%f_es=%f_en=%f_kl=%f" % (
        num_workers,
        vf_loss_coeff,
        entropy_coeff_start,
        entropy_coeff_end,
        kl_coeff
    )

    experiment_name = "{0}_{1}_{2}".format("PPO", layout_name, params_str)

    # Rewards the agent will receive for intermediate actions
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 0,
        "DISH_PICKUP_REWARD": 0,
        "SOUP_PICKUP_REWARD": 0,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }

    # Max episode length
    horizon = 400

    # Constant by which shaped rewards are multiplied by when calculating total reward
    reward_shaping_factor = 0.0

    # Linearly anneal the reward shaping factor such that it reaches zero after this number of timesteps
    reward_shaping_horizon = float('inf')

    # To be passed into rl-lib model/custom_options config
    model_params = {
        "use_lstm" : use_lstm,
        "NUM_HIDDEN_LAYERS" : NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS" : SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS" : NUM_FILTERS,
        "NUM_CONV_LAYERS" : NUM_CONV_LAYERS,
        "CELL_SIZE" : CELL_SIZE,
        "D2RL": D2RL
    }

    # to be passed into the rllib.PPOTrainer class
    training_params = {
        "num_workers" : num_workers,
        "train_batch_size" : train_batch_size,
        "sgd_minibatch_size" : sgd_minibatch_size,
        "rollout_fragment_length" : rollout_fragment_length,
        "num_sgd_iter" : num_sgd_iter,
        "lr" : lr,
        "lr_schedule" : lr_schedule,
        "grad_clip" : grad_clip,
        "gamma" : gamma,
        "lambda" : lmbda,
        "vf_share_layers" : vf_share_layers,
        "vf_loss_coeff" : vf_loss_coeff,
        "kl_coeff" : kl_coeff,
        "clip_param" : clip_param,
        "num_gpus" : num_gpus,
        "seed" : seed,
        "evaluation_interval" : evaluation_interval,
        "entropy_coeff_schedule" : [(0, entropy_coeff_start), (entropy_coeff_horizon, entropy_coeff_end)],
        "eager" : eager,
        "log_level" : "WARN" if verbose else "ERROR"
    }

    # To be passed into AgentEvaluator constructor and _evaluate function
    evaluation_params = {
        "ep_length" : evaluation_ep_length,
        "num_games" : evaluation_num_games,
        "display" : evaluation_display
    }

    environment_params = {
        # To be passed into OvercookedGridWorld constructor

        "mdp_params" : {
            "layout_name": layout_name,
            "rew_shaping_params": rew_shaping_params
        },
        # To be passed into OvercookedEnv constructor
        "env_params" : {
            "horizon" : horizon
        },

        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params" : {
            "reward_shaping_factor" : reward_shaping_factor,
            "reward_shaping_horizon" : reward_shaping_horizon,
            "use_phi" : use_phi,
            # customized reward calculation
            "custom_reward_func": reward_func
        }
    }

    ray_params = {
        "custom_model_id" : "MyPPOModel",
        "custom_model_cls" : RllibLSTMPPOModel if model_params['use_lstm'] else RllibPPOModel,
        "temp_dir" : "/Users/jasmineli/Desktop/tmp" if not GERLACH else "/home/jasmine/moral-ai-irl/result",
        "env_creator" : _env_creator
    }

    params = {
        "model_params" : model_params,
        "training_params" : training_params,
        "environment_params" : environment_params,
        "shared_policy" : shared_policy,
        "num_training_iters" : num_training_iters,
        "evaluation_params" : evaluation_params,
        "experiment_name" : experiment_name,
        "save_every" : save_freq,
        "seeds" : seeds,
        "results_dir" : results_dir,
        "ray_params" : ray_params,
        "verbose" : verbose
    }
    
    return params

def getMAIDummyFE(train_config):
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
    
    left_state = feat_states[left_idx]
    left_state = np.sum(left_state, axis=0)
    right_state = feat_states[right_idx]
    right_state = np.sum(right_state, axis=0)
    
    feat_states = {"leftFE": left_state, "rightFE": right_state}
    return feat_states

def getRLAgentFE(train_config, layout): #get the feature expectations of a new policy using RL agent
    '''
    Trains an RL agent with the current reward function. 
    Then rolls out one trial of the trained agent and calculate the feature expectation of the RL agent.
    - train_config: the configuration taken by the rllib trainer
    
    Returns the feature expectation.
    '''
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
        right_state = feat_states[right_idx]
        right_state = np.sum(right_state, axis=0)
        # print(f'RL right FE shape = {right_state.shape}')
        return right_state
    else:
        assert layout == 'mai_separate_coop_left'
        left_state = feat_states[left_idx]
        left_state = np.sum(left_state, axis=0)
        # print(f'RL left FE shape = {left_state.shape}')
        return left_state


if __name__ == "__main__":
    # inputs, targets, seq_lens = load_data()
    # print(f'input = {inputs.shape}')
    # print(f'targets = {targets.shape}')

    EPSILON = 100
    TRIAL = 2

    # init 
    reward_model = LinearReward(96)
    config = get_train_config(reward_func=reward_model.getRewards)
    expertFE = getMAIDummyFE(config)    # the expert feature expectation (only uses mdp_params and env_params in config)
    layout = config["environment_params"]["mdp_params"]["layout_name"]
    expertFE = expertFE['leftFE'] if layout == 'mai_separate_coop_left' else expertFE['rightFE']
    irl_agent = irlAppAgent(expertFE=expertFE)

    i = 0
    while True:
        print(f'----------------  {i}  ----------------')
        config = get_train_config(reward_func=reward_model.getRewards)
        agentFE = getRLAgentFE(config, layout=layout)
        # print(agentFE)
        W, currentT = irl_agent.optimalWeightFinder(agentFE, reward_model.getRewards)
        
        W = W.reshape((-1, 1))
        assert reward_model.weights.shape == W.shape
        reward_model.updateWeights(W)

        if currentT <= EPSILON:
            break
        i += 1
        


    # TODO: log the weights


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
