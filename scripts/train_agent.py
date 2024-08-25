import gymnasium as gym
from stable_baselines3 import PPO

import lemon_env

if __name__ == '__main__':
    env = gym.make('LemonEnv-v0')

    # Initialize the model, specifying GPU usage
    model = PPO("CnnPolicy", env, verbose=1, n_steps=256, n_epochs=4)

    # Train the model
    model.learn(total_timesteps=1024)

    # Save the model
    model.save("ppo_lemon_env")
