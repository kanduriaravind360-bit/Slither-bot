from env import slitherenv

env = slitherenv()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="slither"
)
model = PPO(
    "CnnPolicy",
    env,
    device="cuda",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.03,
    verbose=1,
)
model.learn(total_timesteps=1000000, callback=checkpoint_callback)
model.save("Slitherppo_v1.0")
