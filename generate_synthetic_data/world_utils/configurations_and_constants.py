import numpy as np

from mujoco_env.tasks.null_task import NullTask

cfg = dict(
    scene=dict(
        resource='lemons',
    ),
    robot=dict(
        resource='xarm7',
        privileged_info=True,
    ),
    task=NullTask,
)

frame_skip = 5

ORIGIN = [0., 0., 0.3]

num_cameras = 4
