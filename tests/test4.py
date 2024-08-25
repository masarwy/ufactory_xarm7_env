from lemon_picking_world.lemons_world import LemonWorld

if __name__ == '__main__':
    world = LemonWorld()

    N_EPISODES = 1
    N_STEPS = 200

    for _ in range(N_EPISODES):
        _, _ = world.reset()

        for _ in range(N_STEPS):
            action = world.action_space.sample()
            world.step(action)
    world.close()
