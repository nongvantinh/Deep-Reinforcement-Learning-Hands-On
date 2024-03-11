import gymnasium


if __name__ == "__main__":
    env = gymnasium.make("CartPole-v0", render_mode="human")
    # env = gymnasium.wrappers.Monitor(env, "recording")

    total_reward = 0.0
    total_steps = 0
    observation, info = env.reset()

    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        if terminated or truncated:
            break

    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
    env.close()
    env.env.close()
