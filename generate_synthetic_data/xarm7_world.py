from copy import deepcopy

from mujoco_env import MujocoEnv
from .world_utils.configurations_and_constants import *
from .world_utils.camera import Camera
from motion_planning.xarm7_ik import PyBulletIKSolver


class XArmWorld:
    def __init__(self, render_mode='human', cfg=cfg):
        self.render_mode = render_mode
        self._env = MujocoEnv.from_cfg(cfg=cfg, render_mode=render_mode, frame_skip=frame_skip)
        obs, info = self._env.reset()  # once, for info, later again
        self._mj_model = info['privileged']['model']
        self._mj_data = info['privileged']['data']

        self._env_entity = self._env.agent.entity

        self.robot_joint_pos = None  # will be updated in reset
        self.robot_joint_velocities = None  # --""--
        self.ee_pose = None

        self.ik_solver = PyBulletIKSolver(urdf_path='../motion_planning/assets/xarm7_robot.urdf')
        self.dt = self._mj_model.opt.timestep * frame_skip

        self.cameras = []
        for i in range(num_cameras):
            specs = {
                'height': 240,
                'width': 320,
                'dist': 1.4,
                'elevation': -40,
                'azimuth': i * 360. / num_cameras,

            }
            self.cameras.append(Camera(self._mj_model, self._mj_data, specs))

        self.reset()

    def reset(self):

        obs, _ = self._env.reset()
        self.robot_joint_pos = obs['robot_state'][:7]
        self.robot_joint_velocities = obs["robot_state"][13:20]
        self.ee_pose = obs["end_effector_pose"]

        self.step(self.robot_joint_pos)

        if self.render_mode == "human":
            self._env.render()

        return self.get_state()

    def step(self, target_joint_pos):

        self._env_step(target_joint_pos)

        if self.render_mode == "human":
            self._env.render()

        return self.get_state()

    def simulate_steps(self, n_steps):
        """
        simulate n_steps in the environment without moving the robot
        """
        config = self.robot_joint_pos
        for _ in range(n_steps):
            self.step(config)

    def render(self):
        return self._env.render()

    def get_state(self):
        state = {"robot_joint_pos": self.robot_joint_pos,
                 "robot_joint_velocities": self.robot_joint_velocities,
                 'ee_pose': self.ee_pose}

        return deepcopy(state)

    def move_to(self, target_pos, tolerance=0.05, end_vel=0.1, max_steps=None):
        """
        move robot joints to target config, until it is close within tolerance, or max_steps exceeded.
        @param target_pos: position to move to
        @param tolerance: distance withing configuration space to target to consider as reached
        @param max_steps: maximum steps to take before stopping
        @return: state, success

        success is true if reached goal within tolerance, false otherwise
        """

        target_pos_cpy = target_pos.copy()
        target_pos_cpy[2] -= 0.12  # the Robot is defined 0.12 units above the floor
        target_joint_pos = self.ik_solver.compute_ik(end_effector_index=7, target_position=target_pos)

        step = 0
        data = {}
        while np.linalg.norm(self.robot_joint_pos[:7] - target_joint_pos) > tolerance \
                or np.linalg.norm(self.robot_joint_velocities) > end_vel:
            if max_steps is not None and step > max_steps:
                return self.get_state(), False

            data[step] = {}

            state = self.step(target_joint_pos)
            data[step]['robot_joints'] = state['robot_joint_pos']
            data[step]['end_effector_pose'] = state['ee_pose']

            for i, camera in enumerate(self.cameras):
                data[step][f'camera_{i}'] = camera()

            step += 1

        return data, True

    def _env_step(self, target_joint_pos):
        """ run environment step and update state of self accordingly"""

        action = np.concatenate((target_joint_pos, [0]))

        obs, r, term, trunc, info = self._env.step(action)

        self.robot_joint_pos = obs['robot_state'][:7]
        self.robot_joint_velocities = obs['robot_state'][13:20]
        self.ee_pose = obs["end_effector_pose"]
