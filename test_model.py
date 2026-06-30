from env import slitherenv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
env = DummyVecEnv([
    lambda: Monitor(slitherenv())
])
env = VecFrameStack(env, 4)
from stable_baselines3 import PPO
policy_kwargs = dict(
    net_arch=[512, 512, 256]
)
model = PPO.load(
    "checkpoints/slither_1000000_steps.zip",
    env=env,
    device="cuda",
    policy_kwargs=policy_kwargs,
)
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()