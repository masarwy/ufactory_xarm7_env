from generate_synthetic_data.xarm7_world import XArmWorld

if __name__ == '__main__':
    world = XArmWorld()
    data, _ = world.move_to([0.5, 0.5, 0.2])
    print('ok')
