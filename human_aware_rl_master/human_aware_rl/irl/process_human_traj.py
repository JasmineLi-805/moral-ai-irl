import json
import glob
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action

"""
run command: `python process_human_traj.py`
"""


def load_json(data_dir):
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
                        state = a_participant['game_rounds'][0]['data']['trajectory'][j]['state']
                        # result.append(state)
                        result.append(OvercookedState.from_dict(state))
                    results.append(result)
                    prev_sample = i
    else:
        return None
    print(f'filtered out {len(results)} trajectories')
    return results, gridworld


data_dir = '/home/jasmine/moral-ai-irl/overcooked_participants_data/'
games = load_json(data_dir)
remove_idle_states(games)
trajectory, gridworld = filter_trajectory(games)
print(f'printing state strings')

for t in range(len(trajectory)):
    traj = trajectory[t]
    i = 0
    for state in traj:
        print(f'trajectory {t}, time step {i}')
        print(gridworld.state_string(state))
        print()
        i += 1


# JOINT ACTION = a_participant['game_rounds'][0]['data']['trajectory'][0]['joint_action']
# SCORE = a_participant['game_rounds'][0]['data']['trajectory'][0]['score']
