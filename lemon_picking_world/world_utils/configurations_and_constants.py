from mujoco_env.tasks import NullTask

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

task_cfg = dict(
    box_center=dict(
        x=0.3,
        y=-0.3
    ),
    box_dims=dict(
        length=0.2,
        width=0.2
    )
)

frame_skip = 5
