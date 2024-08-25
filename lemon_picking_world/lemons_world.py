from copy import deepcopy
import numpy as np
from typing import Optional
import random

from gymnasium import Env, spaces
import mujoco as mj
from mujoco import MjvCamera

from mujoco_env import MujocoEnv
from .world_utils.configurations_and_constants import *
from motion_planning.xarm7_ik import PyBulletIKSolver
from mujoco_env.tasks import LemonPickingTask


class LemonWorld(Env):
    def __init__(self, render_mode='human', image_height=240, image_width=240, cfg=cfg):
        self.render_mode = render_mode
        self._env = MujocoEnv.from_cfg(cfg=cfg, render_mode=render_mode, frame_skip=frame_skip)
        obs, info = self._env.reset()  # once, for info, later again
        self._mj_model = info['privileged']['model']
        self._mj_data = info['privileged']['data']

        self._env_entity = self._env.agent.entity

        self.im_h, self.im_w = image_width, image_height
        self.action_space = self.action_space = spaces.MultiDiscrete([self.im_h, self.im_w])

        self.robot_joint_pos = None  # will be updated in reset
        self.robot_joint_velocities = None  # --""--

        self.ik_solver = PyBulletIKSolver(urdf_path='../motion_planning/assets/xarm7_robot.urdf')
        self.dt = self._mj_model.opt.timestep * frame_skip

        self.target_box_center = task_cfg['box_center']
        self.target_box_dims = task_cfg['box_dims']

        x_low = self.target_box_center['x'] - self.target_box_dims['length'] / 2
        x_high = self.target_box_center['x'] + self.target_box_dims['length'] / 2
        y_low = self.target_box_center['y'] - self.target_box_dims['width'] / 2
        y_high = self.target_box_center['y'] + self.target_box_dims['width'] / 2

        self.workspace_x_lims = [x_low, x_high]
        self.workspace_y_lims = [y_low, y_high]

        self.home1 = self.ik_solver.compute_ik(end_effector_index=7, target_position=[0.3, 0.3, 0.4])
        self.home2 = self.ik_solver.compute_ik(end_effector_index=7, target_position=[0.3, -0.3, 0.4])

        self.task = LemonPickingTask(self.target_box_center, self.target_box_dims, time_limit=20)

        self.current_gripper = 0.

        self.renderer = mj.Renderer(self._mj_model, self.im_h, self.im_w)
        self.camera = MjvCamera()
        self.camera.lookat = [0.3, 0.3, 0.]
        self.camera.distance = 0.5
        self.camera.elevation = -90
        self.camera.azimuth = 0.

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._env.reset()

        self._mj_model = info['privileged']['model']
        self._mj_data = info['privileged']['data']

        self.robot_joint_pos = obs['robot_state'][:7]
        self.robot_joint_velocities = obs["robot_state"][13:20]

        self.task.reset(self._mj_model, self._mj_data)

        self._step(self.robot_joint_pos)

        if self.render_mode == "human":
            self._env.render()

        self.update_scene()
        ob = self.renderer.render()
        info = self.task.get_info()

        return ob, info

    def _step(self, target_joint_pos):

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
            self._step(config)

    def render(self):
        return self._env.render()

    def get_state(self):
        state = {"robot_joint_pos": self.robot_joint_pos,
                 "robot_joint_velocities": self.robot_joint_velocities,
                 "task_info": self.task.get_info()}

        return deepcopy(state)

    def move_to_conf(self, target_joint_pos, tolerance=0.05, end_vel=0.1, max_steps=None):
        step = 0
        while np.linalg.norm(self.robot_joint_pos[:7] - target_joint_pos) > tolerance \
                or np.linalg.norm(self.robot_joint_velocities) > end_vel:
            if max_steps is not None and step > max_steps:
                return False

            self._step(target_joint_pos)

            step += 1
        return True

    def step(self, action):
        """
        move robot joints to target config, until it is close within tolerance, or max_steps exceeded.
        @param action: position to move above
        @return: state, success

        success is true if reached goal within tolerance, false otherwise
        """

        action_coord = self.pixel2world(action)

        self.move_to_conf(self.home1)

        target_pos = [action_coord[0], action_coord[1], 0.04]
        target_joint_pos = self.ik_solver.compute_ik(end_effector_index=7, target_position=target_pos)
        self.move_to_conf(target_joint_pos)

        self.activate_gripper()

        self.move_to_conf(self.home1)
        self.move_to_conf(self.home2)

        drop_location = [random.uniform(*self.workspace_x_lims), random.uniform(*self.workspace_y_lims), 0.05]
        drop_joint_pos = self.ik_solver.compute_ik(end_effector_index=7, target_position=drop_location)
        self.move_to_conf(drop_joint_pos)

        self.deactivate_gripper()

        self.move_to_conf(self.home2)

        self.update_scene()
        ob = self.renderer.render()
        reward = self.task.score(self._mj_data)
        done = self.task.is_done()
        info = self.task.get_info()

        return ob, reward, done, False, info

    def activate_gripper(self):
        self.current_gripper = 170.
        action = np.concatenate((self.robot_joint_pos[:7], [170.]))
        for _ in range(15):
            self._env.step(action)

    def deactivate_gripper(self):
        self.current_gripper = -170.
        action = np.concatenate((self.robot_joint_pos[:7], [-170.]))
        for _ in range(15):
            self._env.step(action)

    def _env_step(self, target_joint_pos):
        """ run environment step and update state of self accordingly"""

        action = np.concatenate((target_joint_pos, [self.current_gripper]))

        obs, r, term, trunc, info = self._env.step(action)

        self.robot_joint_pos = obs['robot_state'][:7]
        self.robot_joint_velocities = obs['robot_state'][13:20]

    def close(self):
        self._env.close()

    def update_scene(self):
        self.renderer.update_scene(self._mj_data, self.camera)

    def pixel2world(self, action):
        # the origin of the image is at x_high, y_high
        x_high, y_high = 0.4, 0.4
        resolution = self.target_box_dims['width'] / self.im_w
        return [x_high - resolution * action[0], y_high - resolution * action[1]]




