# IRL Design Doc

## Components

- [x] Configure coding environment
- [x] Data preprocessing (data file loading, dataset cleaning)
- [ ] Implement RL training code
  - [x] Set the game environment to use customized layout
  - [x] Add the Dummy agent to the pair of agents playing the game
    - [x] Q: states are featurized before passing to the Dummy agent, even though featurization function is set to `None`, causing type mismatch.
    - [ ] Q: The observation requires the custumized space to have a shape, which is not applicable to the OvercookedState object.
  - [ ] Change the reward evaluation method to customized reward function
    - [x] Add a customized evaluate function where new reward calculation can be inserted
    - [ ] Find a suitable reward function (model structure)
  - [ ] Test the correctness of the code (agents learning, layout correctly loaded, etc.)
  - [ ] Tune the hyperparameters for training.
  - [ ] Estimate the time required for training.
- [ ] IRL implementation
  - [ ] Code that initializes the IRL training environment
  - [ ] Training code using Pytorch and the RL training code
  - [ ] Add util methods for loading and automatically saving the models
  - [ ] Test the correctness of the code (reward function converges, RL agent learning the rewards, etc.)

## RL Training Code Modifications

### Overview

`human_aware_rl_master/human_aware_rl/ppo/ppo_rllib_client.py` contains the training code for a client to train their own agent to play with a trained behavioral cloning agent. Our RL training code can adopt the existing code by making the following modifications:
    1. Change the layout to ours (`mai_separate_coop_left` and `mai_separate_coop_right`).
    2. Change the BC agent to the Dummy agent used in our game.
    3. Add our own reward function to the evaluation function.

### Implementation

#### layout change

The layout can be easily changed in `ppo_rllib_client.py` by setting the `layout_name` paramter.

#### Dummy agent

The existing code uses RLLib to do the training. 
`human_aware_rl_master/human_aware_rl/dummy/rl_agent.DummyPolicy`: the wrapper around Dummy agent for it to be compatible with RLLib. In the training process, the `DummyPolicy` would be used to create an agent that interacts with the game environment.

*Add MAI_Dummy Agent to the training process*
- `rl_agent.mai_dummy_feat_fn`: transforms the game state object into a dictionary with two keys: `player_1_held_obj`and `help_obj`, since RLlib requires the input to the agent to be specified in terms of gym.spaces. These keys are the only values used by MAI_Dummy Agent
- `game.py`: Added conditions to handle inputs transformed by the featurization function above.
- `rllib.py`: Inserted MAI_Dummy as the second agent in the training process.

*Add MAI_Dummy Agent to Evaluation*
- `rllib.gen_trainer_from_params`: MAI_Dummy Agent should be automatically added to the eval process when constructing the trainer.
- `rllib.get_rllib_eval_function`: The featurization function should be set explicitly for the dummy agent, otherwise the featurization function for the ppo agent is used.


#### Customized evaluation function

The `evaluate` function in `human_aware_rl_master/human_aware_rl/rllib/rllib.py` is where the trial rollouts and reward calculation occur. It calls into `get_rllib_eval_function` in which returns an evaluation function. I added the new evaluation function `_evaluate_customized_reward` in `get_rllib_eval_function`, parallel with the existing `_evaluate` function.

- In the future, new evaluation functions can be added to the same place. The client only needs to switch the return value of `get_rllib_eval_function` to switch between different evaluation functions

### Testing

#### Model learns from training

The RL agent should show improvements in rewards in the training process. This can be tested by the following methods:

- [x] The agent can be trained using the final score as the reward
- [ ] The agent can be trained using our customized reward function

#### Code correctness

The following are the things to consider when checking the implementation's correctness:

- Intended layout is loaded
- The `coop_cnt` variables are added to the state and featurized
- Agent initialization (position, state, etc)
- Agent behaviors are valid in the layout (eg. some interactive actions can only be performed in certain positions in a layout)
