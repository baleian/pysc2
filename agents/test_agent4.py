import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment as sc2_env
from pysc2.env import run_loop

from collections import deque


class DQNNet(tf.keras.Model):
    def __init__(self):
        super(__class__, self).__init__()
        self.embed_conv_layer = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')    # trainable=False 안하는게 더 성능 좋음
        self.conv_layer1 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.conv_layer2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv_layer3 = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.flatten = layers.Flatten()

    def call(self, states):
        states = tf.transpose(states, perm=[0, 2, 1])
        states = tf.one_hot(states, depth=5, axis=-1)
        states = self.embed_conv_layer(states)
        states = self.conv_layer1(states)
        states = self.conv_layer2(states)
        states = self.conv_layer3(states)
        q_value = self.flatten(states)
        return q_value


class DQNAgent(object):

    def __init__(self, train=True, discount_factor=0.99, learning_rate=1e-4):
        self.is_training = train
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay_steps = 10000
        self.max_memory = 10000
        self.memory = deque(maxlen=self.max_memory)
        self.batch_size = 16
        self.train_steps = 0

        self.obs_spec = None
        self.action_spec = None
        self.model = None
        self.target_model = None

        self.last_state = None
        self.last_action = None

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.model = DQNNet()
        self.target_model = DQNNet()

    def reset(self):
        self.last_state = None
        self.last_action = None
        # Update target model
        self.target_model.set_weights(self.model.get_weights())

    def step(self, obs):
        if obs.step_type == sc2_env.StepType.FIRST:
            return sc2_actions.FUNCTIONS.select_army([0])

        state = self.get_state(obs)
        reward = obs.reward
        action = self.get_action(state)
        done = obs.step_type == sc2_env.StepType.LAST

        if self.is_training and self.last_state is not None and self.last_action is not None:
            self.train(self.last_state, self.last_action, reward, state, done)
        self.last_state = state
        self.last_action = action

        return sc2_actions.FUNCTIONS.Move_screen([0], [int(action / 84), action % 84])

    def get_state(self, obs):
        return obs.observation.feature_screen.player_relative

    def predict(self, state):
        return self.model(np.array([state]))

    def get_action(self, state):
        epsilon = max(self.epsilon_min, (self.epsilon_max - ((self.epsilon_max - self.epsilon_min) * self.train_steps / self.epsilon_decay_steps)))
        if epsilon > np.random.rand():
            return np.random.randint(0, 84 * 84)
        else:
            q_value = self.predict(state)
            q_value = q_value[0].numpy()
            return np.argmax(q_value)

    def train(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) > self.batch_size:
            indexes = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, replace=False)
            states = np.array([self.memory[i][0] for i in indexes])
            actions = np.array([self.memory[i][1] for i in indexes])
            rewards = np.array([self.memory[i][2] for i in indexes])
            next_states = np.array([self.memory[i][3] for i in indexes])
            dones = np.array([self.memory[i][4] for i in indexes])

            model_params = self.model.trainable_variables
            with tf.GradientTape() as tape:
                q_values = self.model(states)
                one_hot_actions = tf.one_hot(actions, 84 * 84)
                predicts = tf.reduce_sum(one_hot_actions * q_values, axis=1)

                target_q_values = self.target_model(next_states)
                target_q_values = tf.stop_gradient(target_q_values)

                max_q_values = np.amax(target_q_values, axis=-1)
                targets = rewards + (1 - dones) * self.discount_factor * max_q_values

                loss = tf.reduce_mean(tf.square(targets - predicts))
            grads = tape.gradient(loss, model_params)
            self.optimizer.apply_gradients(zip(grads, model_params))
            self.train_steps += 1


import importlib

from absl import app
from absl import flags

FLAGS = flags.FLAGS

def main(unused_argv):
    with sc2_env.SC2Env(
            map_name='CollectMineralShards',
            players=[sc2_env.Agent(sc2_env.Race['terran'])],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
            ),
            visualize=True,
            realtime=False,
    ) as env:
        run_loop.run_loop([DQNAgent()], env)


if __name__ == '__main__':
    app.run(main)
