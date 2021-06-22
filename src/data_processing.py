
import pandas as pd
import json
import sys
sys.path.append('/Users/jasmineli/Desktop/moral-ai-irl')
from human_aware_rl_master.human_aware_rl.human.data_processing_utils import joint_df_trajs_to_overcooked_single
from human_aware_rl_master.human_aware_rl.human.process_dataframes import *
from typing_extensions import final
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from overcooked_ai_py.mdp.actions import Action
import os
import pickle
import numpy as np


CLEAN_DATA_DIR = '/Users/jasmineli/Desktop/moral-ai-irl/data/cleaned'

# the schema for the raw trials, used by convert_json_to_trials
SCHEMA = [
    'state', 'joint_action', 'time_left', 'split_score', 'time_elapsed', 'cur_gameloop', 'layout',
    'layout_name', 'trial_id', 'player_0_id', 'player_1_id', 'player_0_is_human', 'player_1_is_human',
    'agent_coop_count', 'country', 'city', 'locale', 'id'
]

# The feature name match from schema to raw json file
SCHEMA_TO_JSON_KEY = {
    'state': 'state',
    'joint_action': 'joint_action',
    'time_left': 'time_left',
    'split_score': 'score',
    'time_elapsed': 'time_elapsed',
    'cur_gameloop': 'cur_gameloop',
    'layout': 'layout',
    'layout_name': 'layout_name',
    'id': 'litw_uuid',                 # unique to each complete play
    # unique within a complete play (3 trials per play)
    'trial_id': 'name',
    'player_0_id': 'player_0_id',
    'player_1_id': 'player_1_id',
    'player_0_is_human': 'player_0_is_human',
    'player_1_is_human': 'player_1_is_human',
    'agent_coop_count': 'agent_coop_count',
    'country': 'country',
    'city': 'city',
    'locale': 'locale'
}


#####################
# LOW LEVEL METHODS #
#####################

def read_json_to_trials(file_path, verbose=False):
    '''
    Convert the initial json file to a list of trials.
    Returns: trials
        - trials: a list of trials.
        - trials[i]: a single trial with len(trial[i]) timesteps.
        - trials[i][j]: a dictionary formatted as SCHEMA
    '''
    trials = []
    with open(file_path, 'r') as f:
        if verbose:
            print(f'reading raw json file: {file_path}.')
        data = json.load(f)
        for game in data['games']:
            trial = []
            for step in game['data']['trajectory']:
                tr = {}
                for s in SCHEMA:
                    json_key = SCHEMA_TO_JSON_KEY[s]
                    if json_key in data:
                        tr[s] = data[json_key]
                    elif json_key in game:
                        tr[s] = game[json_key]
                    elif json_key in game['data']:
                        tr[s] = game['data'][json_key]
                        if s == 'split_score':
                            tr[s] = step[json_key]
                    elif json_key in step:
                        tr[s] = step[json_key]
                    else:
                        sys.exit('schema name mismatch')
                assert(len(tr) == len(SCHEMA))
                trial.append(tr)
            if verbose:
                trial_id = trial[0]['trial_id']
                print(f'  trial {trial_id}: loaded {len(trial)} timesteps.')

            # add splitted reward on each state transition
            final_step = trial[-1]
            final_step['split_reward'] = [0, 0]
            for i in range(0, len(trial) - 1):
                cur_score = trial[i]['split_score']
                next_score = trial[i + 1]['split_score']
                trial[i]['split_reward'] = [next_score[0] -
                                            cur_score[0], next_score[1] - cur_score[1]]

            # add combined reward and score on each state transition
            total_reward = 0
            for i in range(len(trial)):
                trial[i]['score'] = sum(trial[i]['split_score'])
                trial[i]['reward'] = sum(trial[i]['split_reward'])
                total_reward += trial[i]['reward']
            assert(trial[-1]['score'] == total_reward)

            trials.append(trial)
        if verbose:
            game_id = trials[0][0]['id']
            print(f'game {game_id}: loaded {len(trials)} trials.')
    return trials


def convert_trials_to_formatted_df(trials, verbose=False):
    '''
    Converts a list of trials to pandas dataframes, adds 
    '''

    def json_action_to_python_action(action):
        if type(action) is list:
            action = tuple(action)
        if type(action) is str:
            action = action.lower()
        assert action in Action.ALL_ACTIONS
        return action

    def is_interact(joint_action):
        joint_action = tuple(json_action_to_python_action(a)
                             for a in joint_action)
        return np.array([int(joint_action[0] == Action.INTERACT), int(joint_action[1] == Action.INTERACT)])

    def is_button_press(joint_action):
        joint_action = tuple(json_action_to_python_action(a)
                             for a in joint_action)
        return np.array([int(joint_action[0] != Action.STAY), int(joint_action[1] != Action.STAY)])

    def _add_interactivity_metrics(trials):
        # this method is non-destructive
        trials = trials.copy()

        # whether any human INTERACT actions were performed
        def is_interact_row(row): return int(np.sum(np.array(
            [row['player_0_is_human'], row['player_1_is_human']]) * is_interact(row['joint_action'])) > 0)

        # Whehter any human keyboard stroked were performed
        def is_button_press_row(row): return int(np.sum(np.array(
            [row['player_0_is_human'], row['player_1_is_human']]) * is_button_press(row['joint_action'])) > 0)

        # temp column to split trajectories on INTERACTs
        trials['interact'] = trials.apply(is_interact_row, axis=1).cumsum()
        trials['dummy'] = 1
        # Temp column indicating whether current timestep required a keyboard press
        trials['button_press'] = trials.apply(is_button_press_row, axis=1)
        # Add 'button_press_total' column to each game indicating total number of keyboard strokes
        trials = trials.join(trials.groupby(['trial_id'])['button_press'].sum(), on=[
                             'trial_id'], rsuffix='_total')
        # Count number of timesteps elapsed since last human INTERACT action
        trials['timesteps_since_interact'] = trials.groupby(['interact'])[
            'dummy'].cumsum() - 1
        # Drop temp columns
        trials = trials.drop(columns=['interact', 'dummy'])

        return trials

    df = []
    for tr in trials:
        tr = pd.DataFrame(tr)

        # add a new column for total timesteps/gameloops
        tr = tr.join(tr.groupby(['trial_id'])['cur_gameloop'].count(), on=[
                     'trial_id'], rsuffix='_total')
        assert (tr['cur_gameloop_total'][0] == len(tr))

        # Calculate total score for each round
        tr = tr.join(tr.groupby(['trial_id'])['score'].max(), on=[
                     'trial_id'], rsuffix='_total')

        # Calculate button presses
        tr = _add_interactivity_metrics(tr)
        tr['button_presses_per_timstep'] = tr['button_press_total'] / \
            tr['cur_gameloop_total']

        df.append(tr)
    if verbose and len(df) > 0:
        print(f'trials converted to formatted dataframes.')
    return df

######################
# HIGH LEVEL METHODS #
######################

    ######################
    # Data Preprocessing #
    ######################


def json_to_df_pickle(in_file_path, out_file_path, verbose=False):
    """
    High level function that reads the raw json file and converts it to 
    processed pandas dataframe.
    """
    # reads and extract necessary information from raw json file
    trials = read_json_to_trials(in_file_path, verbose=verbose)
    # convert trials to dataframe and add featuers from calculation
    dataframes = convert_trials_to_formatted_df(trials, verbose=verbose)
    dataframes = pd.concat(dataframes)
    ###
    # TODO: Add data filtering here if needed.
    ###
    with open(out_file_path, 'wb') as f:
        pickle.dump(dataframes, file=f)

    if verbose:
        print(f'dataframes of trials written to {out_file_path}')
    return dataframes

    ####################
    # Learning / Train #
    ####################


def get_vectorized_trajectories(layout, data_path, country=None, city=None, verbose=False):
    """
    Get vectorized trajectories for a layout, can specify the city and country of the player.

    Arguments:
        layouts (str): the layout we wish to retrieve data for
        data_path (str): Full path to pickled DataFrame we wish to load.
    """

    def _pad(sequences, maxlen=None, default=0):
        if not maxlen:
            maxlen = max([len(seq) for seq in sequences])
        for seq in sequences:
            pad_len = maxlen - len(seq)
            seq.extend([default]*pad_len)
        return sequences

    def format(inputs, targets):
        # sequence matters, pad the shorter ones with init state
        seq_lens = np.array([len(seq) for seq in inputs])
        seq_padded = _pad(inputs, default=np.zeros((len(inputs[0][0],))))
        targets_padded = _pad(targets, default=np.zeros(1))
        inputs_t = np.dstack(seq_padded).transpose((2, 0, 1))
        targets_t = np.dstack(targets_padded).transpose((2, 0, 1))
        return inputs_t, targets_t, seq_lens

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Tried to load human data from {} but file does not exist!".format(data_path))

    if verbose:
        print("Loading data from {}".format(data_path))

    params = {
        "layouts": [layout],
        "check_trajectories": False,
        "featurize_states": True,
        "data_path": data_path,
        "country": country,
        "city": city,
        "verbose": verbose
    }

    main_trials = pd.read_pickle(data_path)

    trajs, info = joint_df_trajs_to_overcooked_single(
        main_trials,
        **params
    )
    
    states = trajs['ep_states']
    actions = trajs['ep_actions']
    inputs, outputs, seq_len = format(inputs=states, targets=actions)

    return (inputs, outputs, seq_len), trajs, info


if __name__ == "__main__":
    # Processes the raw JSON file and save as pickle file in CLEAN_DATA_DIR
    # in_file_path = '/Users/jasmineli/Desktop/moral-ai-irl/data/study_data.json'
    out_file_path = os.path.join(CLEAN_DATA_DIR, 'study_data_clean.pickle')
    # df = json_to_df_pickle(in_file_path, out_file_path, verbose=True)

    # with open(out_file_path, 'rb') as f:
    #     frame = pickle.load(f)
    #     assert(len(frame) == len(df))
    #     assert(type(frame) == type(df))

    data, traj, info = get_vectorized_trajectories(layout="mai_separate_coop_right",
                                             data_path=out_file_path, 
                                             country='United States',
                                             verbose=True)
    x_state = data[0]
    y_action = data[1]
    seq_len = data[2]
    print(x_state.shape)
    print(y_action.shape)
    print(seq_len)
