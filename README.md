# moral-ai-irl

## SETUP

### Start with a NO-GPU version of it:

1. Add both `human_aware` and `overcooked_demo` as part of your project source / python path.
2. Create and start a virtual/conda environment
3. Install dependencies: `python3 -m pip install -r requirements.txt`

### Let's add GPU capabilities:
...

### Trying it out:

1. Access IRL folder: `human_aware_rl_master/human_aware_rl/irl`
2. Copy the `config_model.py` to `config.py`
3. Configure `HOME_DIR` and `TMP_DIR` inside the `config.py`
4. run one iteration of the IRL training: `PYTHONPATH=../../:../../../ python irl_train.py --epochs 1 -t 1`


## Project Structure

The IRL related code are mainly located in the `human_aware_rl_master/human_aware_rl/irl` directory
- `human_aware_rl_master/human_aware_rl/irl/irl_agent.py`: does most of the calculation in the Apprenticeship Algorithm
- `human_aware_rl_master/human_aware_rl/irl/reward_models.py`: contains the reward functions used for the Apprenticeship Algorithm, currently there is only linear models.
- `human_aware_rl_master/human_aware_rl/irl/irl_train.py`: the irl training code, see train instructions below
- `human_aware_rl_master/human_aware_rl/irl/evaluate.py`: evaluates a reward function from the IRL training by training an RL agent with the reward function and displaying the agent trajectory
- `human_aware_rl_master/human_aware_rl/irl/config.py`: sets up the training configuration for `irl_train.py`, while initializing a training process, `irl_train.py` reads the configuration from this file.

### Training Instructions
1. Navigate to the irl directory using `cd human_aware_rl_master/human_aware_rl/irl`
2. Run IRL training with the command `python irl_train.py [--epochs n] [--trial t]`
3. The result is saved to the `result/T[t]` directory in the current directory.

### Evaluation Instruction
1. Navigate to the irl directory using `cd human_aware_rl_master/human_aware_rl/irl`
2. Run `python evaluate.py --checkpoint <path-to-the-checkpoint>`
