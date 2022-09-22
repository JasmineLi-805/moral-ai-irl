import pickle
import matplotlib.pyplot as plt
from human_aware_rl_master.human_aware_rl.irl.reward_models import LinearReward

from human_aware_rl_master.human_aware_rl.irl.evaluate import load_checkpoint

# T34 Depends on code committed #44710e70c894938d00f75d5e66f6fd66554047db
FILE_PATH = 'T34/best.pickle'
checkpoint = load_checkpoint(FILE_PATH)
