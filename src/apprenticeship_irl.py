import sys
sys.path.append('/Users/jasmineli/Desktop/moral-ai-irl')

from human_aware_rl_master.human_aware_rl.human.process_dataframes import *

DEFAULT_DATA_PARAMS = {
    "layouts": ["cramped_room"],
    "check_trajectories": False,
    "featurize_states" : True,
    "data_path": CLEAN_2019_HUMAN_DATA_TRAIN
}

def load_data():
    def _pad(sequences, maxlen=None, default=0):
        if not maxlen:
            maxlen = max([len(seq) for seq in sequences])
        for seq in sequences:
            pad_len = maxlen - len(seq)
            seq.extend([default]*pad_len)
        return sequences
    
    processed_trajs = get_human_human_trajectories(**DEFAULT_DATA_PARAMS, silent=False)
    inputs, targets = processed_trajs["ep_states"], processed_trajs["ep_actions"]
    
    # sequence matters, pad the shorter ones with init state
    seq_lens = np.array([len(seq) for seq in inputs])
    seq_padded = _pad(inputs, default=np.zeros((len(inputs[0][0],))))
    targets_padded = _pad(targets, default=np.zeros(1))
    seq_t = np.dstack(seq_padded).transpose((2, 0, 1))
    targets_t = np.dstack(targets_padded).transpose((2, 0, 1))

    return seq_t, targets_t, seq_lens

if __name__ == "__main__":
    inputs, targets, seq_lens = load_data()
    print(inputs.shape)
    print(targets.shape)