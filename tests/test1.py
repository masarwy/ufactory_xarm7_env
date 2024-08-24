import numpy as np

from mujoco_env import mujoco_env
from mujoco_env.tasks.null_task import NullTask
from mujoco_env.common.xarm7_fk import forward
from motion_planning.xarm7_ik import PyBulletIKSolver

if __name__ == '__main__':
    cfg = dict(
        scene=dict(
            resource='lemons',
        ),
        robot=dict(
            resource='xarm7',
        ),
        task=NullTask,
    )

    env = mujoco_env.MujocoEnv.from_cfg(cfg=cfg, render_mode="human", frame_skip=5)

    N_EPISODES = 1
    N_STEPS = 2000
    ik_solver = PyBulletIKSolver(urdf_path='../motion_planning/assets/xarm7_robot.urdf')

    try:
        for _ in range(N_EPISODES):
            _, _ = env.reset()
            env.render()

            action = np.zeros((1, 8)).flatten()

            action[:7] = ik_solver.compute_ik(end_effector_index=7, target_position=[0.5, 0.5, 0.2])
            print(action)
            print(forward(action[:7]))

            i = 0
            while i < N_STEPS:
                i += 1
                # actions = env.action_space.sample()

                obs, rewards, term, trunc, info = env.step(action)
                env.render()

    except KeyboardInterrupt:
        pass

    env.close()
