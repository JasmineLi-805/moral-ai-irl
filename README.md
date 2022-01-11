# moral-ai-irl

## IRL Training

The IRL related code are mainly located in the `human_aware_rl_master/human_aware_rl/irl` directory
- `human_aware_rl_master/human_aware_rl/irl/irl_agent.py`: does most of the calculation in the Apprenticeship Algorithm
- `human_aware_rl_master/human_aware_rl/irl/reward_models.py`: contains the reward functions used for the Apprenticeship Algorithm, currently there is only linear models.
- `human_aware_rl_master/human_aware_rl/irl/irl_train.py`: the irl training code, see train instructions below
- `human_aware_rl_master/human_aware_rl/irl/evaluate.py`: evaluates a reward function from the IRL training by training an RL agent with the reward function and displaying the agent trajectory
- `human_aware_rl_master/human_aware_rl/irl/config.py`: sets up the training configuration for `irl_train.py`, while initializing a training process, `irl_train.py` reads the configuration from this file.

### Training Instructions
1. Navigate to the irl directory using `cd human_aware_rl_master/human_aware_rl/irl`
2. Run IRL training with the command `python irl_train.py [--epochs n]`
3. The result is saved to the `result/T[trial num]` directory in the current directory, `trial num` is set in `irl_train.py`

### Evaluation Instruction
1. Navigate to the irl directory using `cd human_aware_rl_master/human_aware_rl/irl`
2. Run `python evaluate.py --checkpoint <path-to-the-checkpoint>`
