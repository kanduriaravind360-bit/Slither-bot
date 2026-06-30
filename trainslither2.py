from env2 import slitherenv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack

env2 = DummyVecEnv([
    lambda: Monitor(slitherenv())
])
env2 = VecFrameStack(env2, 4)


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="slither"
)
model = PPO.load(
    "checkpoints/slither_1000000_steps.zip",
    env=env2,
    device='cuda',
)
model.learn(total_timesteps=200000, callback=checkpoint_callback, reset_num_timesteps=False)
model.save("Slitherppo_v2.1")
