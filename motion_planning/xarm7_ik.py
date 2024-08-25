import pybullet as p
import pybullet_data


class PyBulletIKSolver:
    def __init__(self, urdf_path, use_gui=False):
        # Connect to PyBullet
        if use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Set the search path for PyBullet to find URDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the robot model
        self.robot_id = p.loadURDF(urdf_path)

    def compute_ik(self, end_effector_index, target_position):
        # Compute the inverse kinematics
        joint_angles = p.calculateInverseKinematics(self.robot_id, end_effector_index, target_position)
        return joint_angles

    def close(self):
        self.physics_client.close()
