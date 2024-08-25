from typing import Any, Dict

import numpy as np
import random

from gymnasium.core import ActType
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
from mujoco import mjtGeom

from .scoring import count_lemons_in_place


def set_lemon_position(mj_model, mj_data, block_id, position):
    """
    Set the position of a block in the simulation.
    Args:
        block_id: the id of the block to set the position of.
        position: the position to set the block to, position will be in format [x, y ,z].
    """
    joint_name = f"lemon{block_id}_joint"
    joint_id = mj_model.joint(joint_name).id
    pos_adrr = mj_model.jnt_qposadr[joint_id]
    mj_data.qpos[pos_adrr:pos_adrr + 3] = position


def set_all_lemons_positions(mj_model, mj_data, positions):
    """
    Set the positions of all blocks in the simulation.
    Args:
        positions: a list of positions to set the blocks to, positions will be in format [[x, y ,z], ...].
    """
    # set blocks positions
    for i, pos in enumerate(positions):
        set_lemon_position(mj_model, mj_data, i, pos)


def get_lemon_position(mj_data, lemon_id) -> np.ndarray:
    """
    Get the position of a block in the simulation.
    Args:
        lemon_id: the id of the block to get the position of.
    Returns:
        the position of the block in format [x, y ,z].
    """
    return mj_data.joint(lemon_id).qpos[:3]


class LemonPickingTask:
    def __init__(self, box_center: Dict[str, float], box_dimensions: Dict[str, float], time_limit: int):
        self.lemon_count = 0
        self.picked_lemons = 0
        self.lemon_positions = []
        self.step_count = 0
        self.time_limit = time_limit
        self.box_center = box_center
        self.box_dims = box_dimensions

        x_low = 0.3 - box_dimensions['length'] / 2
        x_high = 0.3 + box_dimensions['length'] / 2
        y_low = 0.3 - box_dimensions['width'] / 2
        y_high = 0.3 + box_dimensions['width'] / 2

        self.workspace_x_lims = [x_low, x_high]
        self.workspace_y_lims = [y_low, y_high]

        self.lemon_size = 0.04

        self.task_lemon_names = None
        self.task_lemons = None

    def reset_lemons(self, mj_model, mj_data):
        # manipulated objects have 6dof free joint that must be named in the mcjf.
        all_joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]

        # all bodies that ends with "lemon"
        self.task_lemon_names = [name for name in all_joint_names if name.startswith("lemon")]
        self.task_lemons = {name: mj_model.joint(name) for name in self.task_lemon_names}

        def check_block_collision(new_pos):
            """Tests if new position for block collides with any other block"""
            for pos in lemons_positions:
                pos_np = np.array(pos)
                if np.linalg.norm(new_pos - pos_np) < 2 * self.lemon_size:
                    return True
            lemons_positions.append(list(new_pos))
            return False

        # randomize block positions
        lemons_positions = []
        for _ in range(len(self.task_lemon_names)):
            # generate random position for block
            lemon_location = [random.uniform(*self.workspace_x_lims), random.uniform(*self.workspace_y_lims), 0.03]
            # check if block collides with any other previous new block position
            while check_block_collision(np.array(lemon_location)):
                # generate new random position for block
                lemon_location = [random.uniform(*self.workspace_x_lims), random.uniform(*self.workspace_y_lims), 0.03]
        # set blocks to new positions
        set_all_lemons_positions(mj_model, mj_data, lemons_positions)

    def reset(self, mj_model, mj_data):
        self.reset_lemons(mj_model, mj_data)
        self.lemon_count = len(self.task_lemon_names)
        self.picked_lemons = 0

    def begin_frame(self, action: ActType) -> None:
        return

    def end_frame(self, action: ActType) -> None:
        self.step_count += 1
        self.picked_lemons += 1  # Assuming the grasp always succeed if picking the right pixel

    def score(self, mj_data) -> float:
        locs = np.array(list(self.get_all_block_positions_dict(mj_data).values()))
        score = count_lemons_in_place(
            lemon_positions=locs,
            box_center=self.box_center,
            box_dimensions=self.box_dims
        )
        return float(score) * (self.is_done() and self.picked_lemons == self.lemon_count)

    def is_done(self) -> bool:
        return self.step_count >= self.time_limit or self.picked_lemons == self.lemon_count

    def get_info(self) -> dict[str, Any]:
        info = {}
        info.update({
            'picked_lemons': self.picked_lemons,
            'lemon_count': self.lemon_count,
        })
        return info

    def update_render(self, viewer: WindowViewer, mj_data):
        lemon_poses = self.get_all_block_positions_dict(mj_data)
        for obj_name, pos in lemon_poses.items():
            viewer.add_marker(
                pos=pos['goal_com'],
                size=[0.1] * 3,
                rgba=[0, .9, 0, 0.3],
                type=mjtGeom.mjGEOM_SPHERE,
                label=obj_name
            )

    def get_all_block_positions_dict(self, mj_data) -> Dict[str, np.ndarray]:
        """
        Get the positions of all blocks in the simulation.
        Returns:
            a dictionary of block names to their positions, positions will be in format {name: [x, y ,z], ...}.
        """
        return {name: get_lemon_position(mj_data, self.task_lemons[name].id) for name in self.task_lemon_names}
