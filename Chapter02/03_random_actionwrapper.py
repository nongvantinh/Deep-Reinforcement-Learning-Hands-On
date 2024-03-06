import gymnasium

import random


class RandomActionWrapper(gymnasium.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gymnasium.make("CartPole-v0", render_mode="human"))

    observation, info = env.reset()
    total_reward = 0.0

    while True:
        observation, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        if terminated or truncated:
            break

    print("Reward got: %.2f" % total_reward)
