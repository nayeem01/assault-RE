import gym
# import random
# from stable_baselines import ACER
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
import numpy as np


class Assault:
    def __init__(self, height, width, channels, actions):
        self.height = height
        self.width = width
        self.channels = channels
        self.actions = actions

    def buildModel(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu',
                                     input_shape=(3, self.height, self.width, self.channels)))
        self.model.add(Convolution2D(
            64, (4, 4), strides=(2, 2), activation='relu'))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.actions, activation='linear'))

    def buildAgent(self):
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(
            eps=0.2), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
        memory = SequentialMemory(limit=1000, window_length=3)
        self.dqn = DQNAgent(model=self.model, memory=memory, policy=policy,
                            enable_dueling_network=True, dueling_type='avg',
                            nb_actions=actions, nb_steps_warmup=1000,
                            batch_size=25
                            )
        self.dqn.compile(Adam(learning_rate=1e-4))

    def test(self):
        self.buildModel()
        self.buildAgent()
        self.dqn.load_weights("")
        scores = self.dqn.test(env, nb_episodes=10, visualize=True)
        print(np.mean(scores.history['episode_reward']))


if __name__ == "__main__":

    env = gym.make("ALE/Assault-v5", render_mode='human')

    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    agent = Assault(height, width, channels, actions)
    agent.test()
    # episodes = 5
    # for episode in range(1, episodes+1):
    #     state = env.reset()
    #     done = False
    #     score = 0

    #     while not done:
    #         # env.render(mode='human')
    #         action = random.choice([0, 1, 2, 3, 4, 5])
    #         n_state, reward, done, info = env.step(action)
    #         score += reward
    #     print('Episode:{} Score:{}'.format(episode, score))
    # env.close()
