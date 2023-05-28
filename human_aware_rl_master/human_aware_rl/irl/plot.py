import os
import torch
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append('/home/jasmine/moral-ai-irl')
sys.path.append('/home/jasmine/moral-ai-irl/human_aware_rl_master/')
from human_aware_rl.rllib.utils import get_base_ae

def load_eval_results(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'rb') as file:
        results = pickle.load(file)
    return results

mdp_params = {
    "layout_name": "mai_separate_coop_left",
    "rew_shaping_params": {
        "PLACEMENT_IN_POT_REW": 0,
        "DISH_PICKUP_REWARD": 0,
        "SOUP_PICKUP_REWARD": 0,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }
}
env_params = {
    "horizon" : 15
}
ae = get_base_ae(mdp_params, env_params)
env = ae.env

file_path = '/home/jasmine/moral-ai-irl/human_aware_rl_master/human_aware_rl/irl/result/bot/T4/eval-epoch=25/eval_traj_epoch=25'
save_dir = '/home/jasmine/moral-ai-irl/human_aware_rl_master/human_aware_rl/irl/result/bot/T4/eval-epoch=25'

pack = load_eval_results(file_path)
trajectory = pack['eval_state']
config = pack['config']

feat_trajectory = []
for traj in trajectory:
    feat_traj = []
    for s in traj:    
        feat_traj.append(env.irl_reward_state_encoding(s, None, 0)[0])
    feat_traj = torch.tensor(feat_traj).unsqueeze(dim=0)
    feat_trajectory.append(feat_traj)
feat_trajectory = torch.vstack(feat_trajectory)
feat_trajectory = torch.swapaxes(feat_trajectory, 0, 1)
feat_trajectory = torch.sum(feat_trajectory, dim=1)

time = 0
for step in feat_trajectory: 
    step = torch.reshape(step, (6,5))
    step = step.transpose(0,1)

    save_path = os.path.join(save_dir, f't{time}.png')
    plt.imshow(step, cmap='viridis', interpolation='nearest')
    plt.title(f'Agent Positions at Timestep={time}')
    plt.colorbar()
    for i in range(step.shape[0]):
        for j in range(step.shape[1]):
            val = step[i, j].item()
            plt.text(j, i, round(val, 3), ha="center", va="center", color="w")
    plt.savefig(save_path)
    plt.clf()
    time += 1