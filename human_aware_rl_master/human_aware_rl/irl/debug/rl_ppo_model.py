from human_aware_rl.irl import config
from human_aware_rl.irl.reward_models import LinearReward
from human_aware_rl.rllib.rllib import gen_trainer_from_params

reward_model = LinearReward(num_in_feature=30)
all_config = config.get_train_config(reward_func=reward_model.getRewards)

trainer = gen_trainer_from_params(all_config)
results = trainer.train()
