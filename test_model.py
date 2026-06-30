from env2 import slitherenv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
env2 = DummyVecEnv([
    lambda: Monitor(slitherenv())
])
env2 = VecFrameStack(env2, 4)
from stable_baselines3 import PPO
policy_kwargs = dict(
    net_arch=[512, 512, 256]
)
model = PPO.load(
    "Slitherppo_v2.1",
    env=env2,
    device="cuda",
    policy_kwargs=policy_kwargs,
)
obs = env2.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env2.step(action)
    if done:
        obs = env2.reset()