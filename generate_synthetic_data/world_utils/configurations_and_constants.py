from mujoco_env.tasks.null_task import NullTask

cfg = dict(
    scene=dict(
        resource='empty_world',
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
