import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment as sc2_env


class A2CNet(tf.keras.Model):
    def __init__(self):
        super(__class__, self).__init__()
        self.embed_conv_layer = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv_layer1 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.conv_layer2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv_lstm = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.spatial_conv_layer = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.flatten = layers.Flatten()
        self.policy_x = layers.Dense(64, activation='softmax')
        self.policy_y = layers.Dense(64, activation='softmax')
        self.value = layers.Dense(1)

    def call(self, states):
        states = tf.transpose(states, perm=[0, 2, 1])
        states = tf.one_hot(states, depth=5, axis=-1)
        states = self.embed_conv_layer(states)
        states = self.conv_layer1(states)
        states = self.conv_layer2(states)
        states = self.conv_lstm(tf.reshape(states, [1, -1, 64, 64, 32]))
        spatial = self.spatial_conv_layer(states)
        spatial_flat = self.flatten(spatial)
        policy_x = self.policy_x(spatial_flat)
        policy_y = self.policy_y(spatial_flat)
        value = self.value(self.flatten(states))
        return policy_x, policy_y, value


class A2CAgent(object):

    def __init__(self, train=True, discount_factor=0.95, learning_rate=1e-4, step_size=40):
        self.is_training = train
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

        self.step_size = step_size
        self.step_memory = []

        self.obs_spec = None
        self.action_spec = None
        self.model = None
        self.last_state = None
        self.last_action = None

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.model = A2CNet()

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

        return sc2_actions.FUNCTIONS.Move_screen([0], [action[0], action[1]])

    def get_state(self, obs):
        return obs.observation.feature_screen.player_relative

    def get_action(self, state):
        policy_x, policy_y, _ = self.model(np.array([state]))
        policy_x = policy_x[0].numpy()
        policy_y = policy_y[0].numpy()
        x = np.random.choice(len(policy_x), 1, p=policy_x)[0]
        y = np.random.choice(len(policy_y), 1, p=policy_y)[0]
        return [x, y]

    def calculate_G(self, rewards, next_state):
        G = np.zeros_like(rewards, dtype='float32')
        next_value = 0
        if next_state is not None:
            _, _, next_value = self.model(np.array([next_state]))
            next_value = next_value[0].numpy()
        for t in reversed(range(0, len(rewards))):
            value = rewards[t] + self.discount_factor * next_value
            G[t] = value
            next_value = value
        return G

    def train(self, state, action, reward, next_state, done):
        self.step_memory.append((state, action, reward))

        if done or (len(self.step_memory) >= self.step_size):
            states = np.array([item[0] for item in self.step_memory])
            actions_x = np.array([item[1][0] for item in self.step_memory])
            actions_y = np.array([item[1][1] for item in self.step_memory])
            rewards = np.array([item[2] for item in self.step_memory])
            targets = self.calculate_G(rewards, None if done else next_state)
            self.step_memory = []
        else:
            return

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            policy_x, policy_y, values = self.model(states)
            targets = tf.convert_to_tensor(targets[:, None], dtype=tf.float32)

            one_hot_actions_x = tf.one_hot(actions_x, 64, axis=1)
            one_hot_actions_y = tf.one_hot(actions_y, 64, axis=1)
            action_probs_x = tf.reduce_sum(one_hot_actions_x * policy_x, axis=1, keepdims=True)
            action_probs_y = tf.reduce_sum(one_hot_actions_y * policy_y, axis=1, keepdims=True)
            action_probs = action_probs_x * action_probs_y
            cross_entropy = -tf.math.log(tf.clip_by_value(action_probs, 1e-12, 1.))

            policy = tf.expand_dims(policy_x, -1) * tf.expand_dims(policy_y, 1)
            entropy = -tf.reduce_sum(policy * tf.math.log(tf.clip_by_value(policy, 1e-12, 1.)), axis=1)

            advantages = targets - values

            actor_loss = tf.reduce_sum(cross_entropy * tf.stop_gradient(advantages))
            critic_loss = 0.5 * tf.reduce_sum(tf.square(advantages))
            entropy = 0.001 * -tf.reduce_sum(entropy)

            loss = actor_loss + critic_loss + entropy

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
