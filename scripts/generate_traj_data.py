import os
import pickle
import datetime
import random

from generate_synthetic_data.xarm7_world import XArmWorld


def generate_random_pos():
    x1 = random.uniform(-0.5, 0.5)
    x2 = random.uniform(-0.5, 0.5)
    x3 = random.uniform(0, 0.3)
    return [x1, x2, x3]


if __name__ == '__main__':
    world = XArmWorld()

    folder_path = '../data/pose_estimation'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    num_trajectories = 2

    for _ in range(num_trajectories):
        file_path = os.path.join(folder_path, f'{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl')

        target_pos = generate_random_pos()

        world.reset()
        data, _ = world.move_to(target_pos)

        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
