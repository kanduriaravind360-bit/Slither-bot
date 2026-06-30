from env import slitherenv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack

env = DummyVecEnv([
    lambda: Monitor(slitherenv())
])
env = VecFrameStack(env, 4)


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="slither"
)
policy_kwargs = dict(
    net_arch=[512, 512, 256]
)
model = PPO(
    "MlpPolicy",
    env,
    device="cuda",
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=2048,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)
model.learn(total_timesteps=1000000, callback=checkpoint_callback)
model.save("Slitherppo_v2.0")
