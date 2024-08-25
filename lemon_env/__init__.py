from gymnasium.envs.registration import register
from .lemon_env import LemonEnv

register(
    id='LemonEnv-v0',
    entry_point='lemon_env.lemon_env:LemonEnv',
    max_episode_steps=1000,
)
