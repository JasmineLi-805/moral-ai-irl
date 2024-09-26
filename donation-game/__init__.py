from gymnasium.envs.registration import register

register(
    id="gym/GridWorld-v0",
    entry_point="envs:GridWorldEnv",
)

register(
    id="gym/DonationGame-v0",
    entry_point="envs:DonationGameEnv",
)
