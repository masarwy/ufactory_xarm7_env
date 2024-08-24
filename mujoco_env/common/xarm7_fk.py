import numpy as np

from .transform import Transform

# xArm 7 dh parameters
# This can be find here:
# https://help.ufactory.cc/en/articles/4330809-kinematic-and-dynamic-parameters-of-ufactory-xarm-series
d = [0.267, 0.0, 0.293, 0.0, 0.3425, 0.0, 0.097]
a = [0.0, 0.0, 0.0525, 0.0775, 0.0, 0.076, 0.]
alpha = [-np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, 0.]


# Forward Kinematics Function
def forward(joint_angles: np.ndarray) -> Transform:
    T = np.eye(4)

    for i in range(len(d)):
        A = np.array([
            [np.cos(joint_angles[i]), -np.sin(joint_angles[i]) * np.cos(alpha[i]),
             np.sin(joint_angles[i]) * np.sin(alpha[i]), a[i] * np.cos(joint_angles[i])],
            [np.sin(joint_angles[i]), np.cos(joint_angles[i]) * np.cos(alpha[i]),
             -np.cos(joint_angles[i]) * np.sin(alpha[i]), a[i] * np.sin(joint_angles[i])],
            [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
            [0, 0, 0, 1]
        ])

        T = np.dot(T, A)

    return Transform.from_matrix(T)
