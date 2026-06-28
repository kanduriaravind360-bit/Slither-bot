from env import slitherenv
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: slitherenv()])
from stable_baselines3 import PPO
model = PPO.load(
    "checkpoints/slither_1851472_steps.zip",
    env=env,
    device="cuda",
)
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()