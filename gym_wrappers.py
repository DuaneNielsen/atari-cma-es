from gym import Wrapper

class StepReward(Wrapper):
    def __init__(self, env, step_reward=1):
        super(StepReward, self).__init__(env)
        self.step_reward = step_reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        reward = reward + self.step_reward

        return observation, reward, done, info

    def reset(self):
        return self.env.reset()


class StepOnlyReward(Wrapper):
    def __init__(self, env, step_reward=1):
        super(StepOnlyReward, self).__init__(env)
        self.step_reward = step_reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        reward = self.step_reward

        return observation, reward, done, info

    def reset(self):
        return self.env.reset()