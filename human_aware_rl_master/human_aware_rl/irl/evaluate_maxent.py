import torch
from torch import nn
import pickle
import argparse
import sys, os

from human_aware_rl.ppo.ppo_rllib_client import run
from human_aware_rl_master.human_aware_rl.irl.reward_models import TorchLinearReward
from human_aware_rl.rllib.utils import get_base_ae
from overcooked_ai_py.agents.agent import AgentPair

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_checkpoint(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'rb') as file:
        checkpoint = pickle.load(file)
    return checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate maxent model')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the checkpoint file')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save the results')
    args = parser.parse_args()
    assert args.checkpoint, "commandline argument missing: checkpoint"
    assert args.save_dir, "commandline argument missing: save_dir"
    return args.checkpoint, args.save_dir

if __name__ == "__main__":
    print(f'use gpu: {torch.cuda.is_available()}; device={device}')

    print(f'parsing commandline arguments...')
    checkpoint, save_dir = parse_args()

    # create the directory to save the results
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print(f'initiating...')
    print(f'\tloading model checkpoint from {checkpoint}...')
    checkpoint = load_checkpoint(checkpoint)
    
    print(f'\tretrieving reward model and optimizer...')
    reward_model = checkpoint["reward_model"]
    
    print(f'\tloading configurations...')
    config = checkpoint['config']
    irl_config = config['irl_params']
    config['environment_params']['multi_agent_params']['custom_reward_func'] = reward_model.get_rewards
    print(f'complete')
    
    print(f'start evalutating')
    config['evaluation_params']['display'] = True
    results = None
    while True:
        try:
            results = run(config)
            break
        except Exception as e:
            print(e)
    game_state = results['evaluation']['states']

    pack = {
        'eval_state': game_state,
        'config': config
    }
    epoch = checkpoint['current_epoch']
    file_name = f'eval_traj_epoch={epoch}'
    with open(os.path.join(save_dir, file_name), 'wb') as save_file:
        pickle.dump(pack, save_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'evaluation completed, results saved to {os.path.join(save_dir, file_name)}')