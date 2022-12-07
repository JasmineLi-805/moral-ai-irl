import os
import json
import glob
import pickle
import sys

from matplotlib.pyplot import grid
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from datetime import datetime


def load_human_data_json(data_dir):
    print(f'start data loading, dir={data_dir}')
    games = []
    for f in glob.glob(data_dir + '/*.json'):
        with(open(f, 'r')) as a_file:
            for line in a_file:
                a_participant = json.loads(line)
                try:
                    if a_participant['demographics']['data']['demographics-retake'] == 'no':
                        games.append(a_participant)
                        num_rounds = len(a_participant['game_rounds'])
                        len_traj = len(a_participant['game_rounds'][0]['data']['trajectory'])
                        print(f'game rounds={num_rounds}, traj len={len_traj}')
                except:
                    print('COULD NOT PROCESS: {}'.format(f))
                    pass
    return games

"""
Filters the trajectory and keeps the states where 
one of the two agents is interacting with the 
environment (action != STAY).
"""
def remove_idle_states(trajectories):
    print(f'removing idle states')
    assert trajectories
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
Filters the trajectory and keeps the states where 
the human participant is interacting with the 
environment (action != STAY).
"""
def remove_p0_idle_states(trajectories):
    print(f'removing idle states')
    assert trajectories
    count = 0
    for a_participant in trajectories:
        trajectory_length = len(a_participant['game_rounds'][0]['data']['trajectory'])
        traj = []
        for a_step in a_participant['game_rounds'][0]['data']['trajectory']:
            joint_action = a_step['joint_action']
            if joint_action[0] != [0,0]:  # Action.STAY = [0,0]
                traj.append(a_step)
            else:
                count += 1
        a_participant['game_rounds'][0]['data']['trajectory'] = traj
        print(f'prev traj len={trajectory_length}, curr traj len={len(traj)}')
    print(f'completed removing idle states, removed {count} idle steps')

# TODO: needs to vary the position and orientation based on game configuration and round.
def is_coop_state(a_step, a_trajectory):
    check_state = a_trajectory[a_step]
    if a_step == len(a_trajectory)-1:
        # There is no next step to confirm onion delivery
        return False
    else:
        return check_state['state']['players'][0]['position'] == [4, 3] \
            and check_state['state']['players'][0]['orientation'] == [1, 0] \
            and check_state['state']['players'][0]['held_object'] is not None \
            and check_state['state']['players'][0]['held_object']['name'] == 'onion' \
            and check_state['joint_action'][0] == 'interact' \
            and {'name': 'onion', 'position': [5, 3]} in a_trajectory[a_step + 1]['state']['objects']


def find_soup(state, position):
    if 'objects' in state:
        for obj in state['objects']:
            if obj['name'] == 'soup' and obj['position'] == position:
                return obj
    return None


def is_non_coop_state(a_step, a_trajectory):
    check_state = a_trajectory[a_step]
    current_soup = find_soup(check_state['state'], [3, 0])
    stove_onions_now = 0 if not current_soup else len(current_soup['_ingredients'])
    if a_step == len(a_trajectory)-1:
        # There is no next step to confirm onion delivery
        return False
    else:
        next_state = a_trajectory[a_step+1]
        next_soup = find_soup(next_state['state'], [3, 0])
        stove_onions_next = 0 if not next_soup else len(next_soup['_ingredients'])
        return check_state['state']['players'][0]['position'] == [3, 1] \
            and check_state['state']['players'][0]['orientation'] == [0, -1] \
            and check_state['state']['players'][0]['held_object'] is not None \
            and check_state['state']['players'][0]['held_object']['name'] == 'onion' \
            and check_state['joint_action'][0] == 'interact' \
            and stove_onions_next > stove_onions_now


def get_pick_onion_up_steps(a_trajectory):
    states_of_interest = []
    for step in range(0, len(a_trajectory)):
        if a_trajectory[step]['state']['players'][0]['position'] == [4, 3] \
        and a_trajectory[step]['state']['players'][0]['orientation'] == [0, 1] \
        and a_trajectory[step]['state']['players'][0]['held_object'] is None \
        and a_trajectory[step]['joint_action'][0] == 'interact':
            states_of_interest.append(step)
    return states_of_interest


def get_trajectories_of_interest(starting_steps, the_trajectory, steps_limit=50, is_stop_state=is_coop_state,
                                 stop_overlap=True):
    coop_trajectories = []
    for step_idx in range(len(starting_steps)):
        step = starting_steps[step_idx]
        trial_trajectory = [the_trajectory[step]]
        steps_to_overflow = len(the_trajectory)-step
        if stop_overlap and step_idx < len(starting_steps)-1:
            next_starting_step = starting_steps[step_idx+1]-step
        else:
            next_starting_step = float('inf')
        limit = min(steps_limit, steps_to_overflow, next_starting_step)
        for check_step in range(step+1, step+limit):
            trial_trajectory.append(the_trajectory[check_step])
            if is_stop_state(check_step, the_trajectory):
                trial_trajectory.append(the_trajectory[check_step+1])
                coop_trajectories.append(trial_trajectory)
                break
    return coop_trajectories


TRAJECTORIES_OF_INTEREST = {
    'onion_help': {
        'get_starting_steps': get_pick_onion_up_steps,
        'is_ending_step': is_coop_state
    },
    'onion_cook': {
        'get_starting_steps': get_pick_onion_up_steps,
        'is_ending_step': is_non_coop_state
    }
}


"""
returns a list of filtered trajectories and a GridWorld object.

returns:
 - results: a list of trajectories of interest
 - gridworld: a GridWorld object created from the layout name, can be used in the future.
"""
def filter_trajectory(trajectories, interest='onion_help'):
    if not trajectories:
        print(f'not a trajectory')
        return None
    results = []
    layout_name = trajectories[0]['game_rounds'][0]['data']['layout_name']
    gridworld = OvercookedGridworld.from_layout_name(layout_name)
    if interest in TRAJECTORIES_OF_INTEREST:
        print('WORKING ON {}!'.format(interest))
        for a_participant in trajectories:
            print('CHECKING participant {}...'.format(a_participant['demographics']['data']['user_id']))
            assert a_participant['game_rounds'][0]['data']['layout_name'] == layout_name
            a_trajectory = a_participant['game_rounds'][0]['data']['trajectory']
            get_starting_steps = TRAJECTORIES_OF_INTEREST[interest]['get_starting_steps']
            is_ending_step = TRAJECTORIES_OF_INTEREST[interest]['is_ending_step']
            steps_of_interest = get_starting_steps(a_trajectory)
            trajectories_of_interest = get_trajectories_of_interest(steps_of_interest, a_trajectory, is_stop_state=is_ending_step)
            results = results + trajectories_of_interest
            print('... and found {} trajectories of interest!'.format(len(trajectories_of_interest)))
    else:
        print('interest not in range')
        return None
    print(f'FILTERED out {len(results)} trajectories')
    return results, gridworld


def process_data(data_dir, save_dir, interest):
    games = load_human_data_json(data_dir)
    remove_p0_idle_states(games)
    trajectory, gridworld = filter_trajectory(games, interest)
    pack = {
        'trajectory': trajectory,
        'gridworld': gridworld
    }
    timestamp = datetime.now()
    file_name = 'trajectories_{}_{}.data'.format(interest, timestamp.isoformat('_', 'seconds'))
    with open(os.path.join(save_dir, file_name), 'wb') as save_file:
        pickle.dump(pack, save_file, protocol=4)
    print(f'data file saved to {os.path.join(save_dir, file_name)}')
    with open(os.path.join(save_dir, '{}.json'.format(file_name)), 'w') as save_file:
        save_file.write(json.dumps(trajectory))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        interest = sys.argv[1]
        data_dir = sys.argv[2]
        save_dir = sys.argv[3]
        process_data(data_dir, save_dir, interest)
    else:
        print('USAGE: python process_human_traj.py INTEREST PATH_TO_DATA PATH_TO_OUTPUT')

