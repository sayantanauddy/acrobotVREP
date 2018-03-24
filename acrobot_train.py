"""
Script for training/testing the acrobot controller
"""

import acrobotVREP
import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

ENV_NAME = 'acrobotVREP-v0'
gym.undo_logger_setup()

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env._max_episode_steps = 200
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# Configure and compile the agent using the built-in Keras optimizer 
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Training (comment out the below 2 lines while testing)
agent.fit(env, nb_steps=10000, visualize=True, verbose=0, nb_max_episode_steps=200)

# After training is done, the final weights are saved
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# For using pre-trained weights uncomment below and comment the above two lines (agen.fit and agent.save)
# agent.load_weights('/home/sayantan/computing/repositories/acrobotVREP/ddpg_acrobotVREP-v0_weights.h5f')

# Finally, evaluate our algorithm for 5 episodes.
# agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)

