import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment as sc2_env


class AtariNet(tf.keras.Model):
    def __init__(self):
        super(__class__, self).__init__()
        self.embed_conv_layer = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv_layer1 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.conv_layer2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv_layer3 = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.flatten = layers.Flatten()
        self.state_representation = layers.Dense(256, activation='relu')
        self.value = layers.Dense(1)
        self.policy = layers.Dense(64 * 64, activation='softmax')

    def call(self, states):
        states = tf.transpose(states, perm=[0, 2, 1])
        states = tf.one_hot(states, depth=5, axis=-1)
        states = self.embed_conv_layer(states)
        conv1 = self.conv_layer1(states)
        conv2 = self.conv_layer2(conv1)
        conv3 = self.conv_layer3(conv2)
        state_representation = self.state_representation(self.flatten(conv2))
        value = self.value(state_representation)
        policy = self.policy(self.flatten(conv3))
        return value, policy


class A2CAgent(object):
    def __init__(self, train=True, discount_factor=0.99, learning_rate=1e-4, step_size=10):
        self.is_training = train
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

        self.step_size = step_size
        self.memory = []

        self.obs_spec = None
        self.action_spec = None
        self.model = None
        self.last_state = None
        self.last_action = None

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.model = AtariNet()

    def reset(self):
        self.last_state = None
        self.last_action = None

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

        return sc2_actions.FUNCTIONS.Move_screen([0], [int(action / 64), action % 64])

    def get_state(self, obs):
        return obs.observation.feature_screen.player_relative

    def get_action(self, state):
        _, policy = self.model(np.array([state]))
        policy = policy[0].numpy()
        x = np.random.choice(len(policy), 1, p=policy)[0]
        return x

    def calculate_G(self, rewards, next_state):
        G = np.zeros_like(rewards, dtype='float32')
        next_value = 0
        if next_state is not None:
            next_value, _ = self.predict(next_state)
            next_value = next_value[0].numpy()
        for t in reversed(range(0, len(rewards))):
            value = rewards[t] + self.discount_factor * next_value
            G[t] = value
            next_value = value
        return G

    def _train(self, state, action, target):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            value, policy = self.model(np.array([state]))
            one_hot_action = tf.one_hot([action], 64 * 64, axis=1)
            action_prob = tf.reduce_sum(one_hot_action * policy)
            cross_entropy = -tf.math.log(tf.clip_by_value(action_prob, 1e-12, 1.))
            actor_loss = tf.reduce_mean(cross_entropy * tf.stop_gradient(target - value[0]))
            critic_loss = 0.5 * tf.reduce_mean(tf.square(target - value[0]))
            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

    def train(self, state, action, reward, next_state, done):
        next_value, _ = self.model(np.array([next_state]))
        next_value = next_value[0].numpy()
        target = reward + (1 - done) * self.discount_factor * next_value
        self._train(state, action, target)
        # for t in range(0, len(self.memory)):
        #     self.memory[t][2] += (self.discount_factor ** (len(self.memory) - t)) * reward
        # self.memory.append([state, action, reward])
        #
        # if done:
        #     while len(self.memory) > 0:
        #         sample = self.memory.pop(0)
        #         self._train(sample[0], sample[1], sample[2])
        # elif len(self.memory) >= self.step_size:
        #     next_value, _, _ = self.model(np.array([next_state]))
        #     next_value = next_value[0].numpy()
        #     self.memory[0][2] += (self.discount_factor ** len(self.memory)) * next_value
        #     sample = self.memory.pop(0)
        #     self._train(sample[0], sample[1], sample[2])
