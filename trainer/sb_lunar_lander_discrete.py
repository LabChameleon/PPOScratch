import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("LunarLander-v2", n_envs=4)
env.seed(42)

model = PPO("MlpPolicy", env, verbose=2, policy_kwargs={'ortho_init': True}, seed=42)
model.learn(total_timesteps=1000000)
model.save("ppo_lunarlander")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_lunarlander")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
