import gym
import numpy as np
import random
from gymnasium.envs.registration import register

"""
Remember: register this model with gym in gym-examples/gym_examples/__init__.py
"""
class DonationGameEnv(gym.Env):
    def __init__(
            self, 
            reward_func, 
            player_salary_options: list, 
            robot_salary_options: list , 
            pay_period_options: list, # a list of pay intervals, will randomly pick one at the start of each pay period 
            horizon: int = 100,
            expense_options: list = [1],   # a list of burndown values, will randomly pick one each round. eg: [1, 1, 1, 2] => double % = 25%
            donation_options: list = [1]
            ):
        self.player_salary_options = player_salary_options
        self.robot_salary_options = robot_salary_options
        self.pay_period_options = pay_period_options
        self.expense_options = expense_options
        self.donate_options = donation_options
        
        # set player initial value
        self.X = player_salary_options[0]
        self.Y = robot_salary_options[0]
        self.days_to_payday = pay_period_options[0]

        # the pay period and days to payday
        self.timestep = 1
        self.horizon = horizon

        self.reward_func = reward_func

        self.observation_space = gym.spaces.Dict({
            "X": gym.spaces.Discrete(2**63 - 2),   # the player's amount
            "Y>0": gym.spaces.Discrete(2),           # boolean, whether the robot has money.
            "timestep": gym.spaces.Discrete(2**63 - 2) # the current timestep
        })

        # we have two actions: KEEP or GIVE
        self.action_space = gym.spaces.Discrete(2)
        self.action_map = {
            0: "KEEP",
            1: "GIVE"
        }

        self.give_count = 0
        self.keep_count = 0

        self.donation = 0
        self.actual_transfer = []

    def _get_obs(self):
        robot_has_money = 1 if self.Y > 0 else 0
        return (self.X, robot_has_money, self.timestep)

    def _get_info(self):
        return {
            "timestep": self.timestep,
            "player_amount": self.X,
            "robot_amount": self.Y,
            "pay_period": self.pay_period_options,
            "give_count": self.give_count,
            "keep_count": self.keep_count
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset()

        # set player initial value
        self.X = self.player_salary_options[0]
        self.Y = self.robot_salary_options[0]
        self.days_to_payday = self.pay_period_options[0]
        self.timestep = 1

        self.give_count = 0
        self.keep_count = 0

        self.donation = 0
        self.actual_transfer = []

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # set random values
        player_salary = random.choice(self.player_salary_options)
        robot_salary = random.choice(self.robot_salary_options)
        expense = random.choice(self.expense_options)
        donation = random.choice(self.donate_options)

        # apply player action
        if self.action_map[action] == "KEEP":
            self.keep_count += 1
            if self.X > 0:
                self.actual_transfer.append(-1)
            else:
                self.actual_transfer.append(0)
        elif self.action_map[action] == "GIVE":
            self.give_count += 1
            if self.X > donation:
                self.donation += 1
                self.X -= donation
                self.Y += donation
                self.actual_transfer.append(1)
            else:
                self.actual_transfer.append(0)
        else:
            return Exception("unknown action: {action}")
        
        # calculate the rewards to see if the two players survives the day with their savings
        reward = self.reward_func(self._get_obs())

        self.X = max(self.X - expense, 0)
        self.Y = max(self.Y - expense, 0)

        # The game terminates when the timestep reaches the horizon
        terminated = self.timestep >= self.horizon
        
        # Pay the players at the beginning of the payday.
        self.days_to_payday -= 1
        self.timestep += 1
        if (not terminated and self.days_to_payday == 0):
            self.X += player_salary
            self.Y += robot_salary
            self.days_to_payday = random.choice(self.pay_period_options)

        info = self._get_info()
        observation = self._get_obs()
        return observation, reward, terminated, False, info
