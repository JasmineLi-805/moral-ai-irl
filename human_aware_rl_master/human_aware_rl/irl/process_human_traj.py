import os
import json
import glob
import pickle

from matplotlib.pyplot import grid
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

"""
run command: `python process_human_traj.py`
"""

def load_human_data_json(data_dir):
    games = []
    for f in glob.glob(data_dir + '*.json'):
        with(open(f,'r')) as a_file:
            for line in a_file:
                a_participant = json.loads(line)
                if a_participant['demographics']['data']['demographics-retake']=='no':
                    games.append(a_participant)
                    num_rounds = len(a_participant['game_rounds'])
                    len_traj = len(a_participant['game_rounds'][0]['data']['trajectory'])
                    print(f'game rounds={num_rounds}, traj len={len_traj}')
        # TODO: Remove this to load more than 1 file.
        break
    return games

def remove_idle_states(trajectories):
    print(f'removing idle states')
    count = 0
    for a_participant in trajectories:
        trajectory_length = len(a_participant['game_rounds'][0]['data']['trajectory'])
        traj = []
        for a_step in a_participant['game_rounds'][0]['data']['trajectory']:
            joint_action = a_step['joint_action']
            if joint_action != [[0,0], [0,0]]:  # Action.STAY = [0,0]
                traj.append(a_step)
            else:
                count += 1
        a_participant['game_rounds'][0]['data']['trajectory'] = traj
        print(f'prev traj len={trajectory_length}, curr traj len={len(traj)}')
    print(f'completed removing idle states, removed {count} idle steps')


"""
returns a list of filtered trajectories and a GridWorld object.

returns:
 - results: a list of trajectories in the following format
        [
            [{timestep_1.2 dict}, {timestep_1.2 dict}, ...],
            [{timestep_2.1 dict}, {timestep_2.2 dict}, ...],
            ...
            [{timestep_n.1 dict}, {timestep_n.2 dict}, ...],
        ]
 - gridworld: a GridWorld object created from the layout name, can be used in the future.
"""
def filter_trajectory(trajectories, state='onion_help'):
    if not trajectories:
        return None
    results = []
    layout_name = trajectories[0]['game_rounds'][0]['data']['layout_name']
    gridworld = OvercookedGridworld.from_layout_name(layout_name)
    if state=='onion_help':
        print('filtering onion help')
        for a_participant in trajectories:
            assert a_participant['game_rounds'][0]['data']['layout_name'] == layout_name
            trajectory_length = len(a_participant['game_rounds'][0]['data']['trajectory'])
            prev_sample = -20   # only used to avoid repeated sampling
            for i in range(trajectory_length):
                a_state = a_participant['game_rounds'][0]['data']['trajectory'][i]['state']
                if i - prev_sample >= 10 and a_state['players'][0]['position'] == [4,3]:
                    result = []
                    for j in range(i, min(i + 10, trajectory_length)):
                        # state = a_participant['game_rounds'][0]['data']['trajectory'][j]['state']
                        # action = a_participant
                        # result.append(state)
                        # result.append(OvercookedState.from_dict(state))
                        ts = a_participant['game_rounds'][0]['data']['trajectory'][j]
                        result.append(ts)
                    results.append(result)
                    prev_sample = i
    else:
        return None
    print(f'filtered out {len(results)} trajectories')
    return results, gridworld

if __name__ == "__main__":
    data_dir = '/home/jasmine/moral-ai-irl/overcooked_participants_data/'
    save_dir = '/home/jasmine/moral-ai-irl/overcooked_participants_data/cleaned'

    games = load_human_data_json(data_dir)
    remove_idle_states(games)
    trajectory, gridworld = filter_trajectory(games)
    # state = trajectory[0][0]['state']

    pack = {
        'trajectory': trajectory,
        'gridworld': gridworld
    }
    file_name = f'test_human_1.data'
    with open(os.path.join(save_dir, file_name), 'wb') as save_file:
        pickle.dump(pack, save_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'data file saved to {os.path.join(save_dir, file_name)}')

    # JOINT ACTION = a_participant['game_rounds'][0]['data']['trajectory'][0]['joint_action']
    # SCORE = a_participant['game_rounds'][0]['data']['trajectory'][0]['score']
