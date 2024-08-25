from stable_baselines3 import PPO
import gymnasium as gym

import lemon_env


if __name__ == '__main__':
    env = gym.make('LemonEnv-v0', render_mode='human')

    model = PPO.load("ppo_lemon_env")

    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        env.render()

    env.close()
