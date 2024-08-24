from mujoco_env import mujoco_env
from mujoco_env.tasks.null_task import NullTask


# Define the configuration for multiple robots and tasks
cfg = dict(
    scene=dict(
        resource='xarm7_world',
    ),
    robot=dict(
            resource='xarm7',
        ),
    task=NullTask,
)


# Initialize the environment with the multi-agent configuration
env = mujoco_env.MujocoEnv.from_cfg(cfg=cfg, render_mode="human", frame_skip=5)

N_EPISODES = 1
N_STEPS = 2000

try:
    for _ in range(N_EPISODES):
        _, _ = env.reset()
        env.render()
        i = 0
        while i < N_STEPS:
            i += 1
            actions = env.action_space.sample()

            obs, rewards, term, trunc, info = env.step(actions)
            env.render()
except KeyboardInterrupt:
    pass

env.close()
