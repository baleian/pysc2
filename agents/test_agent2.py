import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features as sc2_features
from pysc2.env import environment as sc2_env


FUNCTIONS = sc2_actions.FUNCTIONS


class SpatialFeaturePreProcessingLayer(layers.Layer):
    def __init__(self, feature_specs):
        super(__class__, self).__init__()
        self.conv_layers = dict()
        self.feature_specs = feature_specs
        for feature_spec in self.feature_specs:
            if feature_spec.type == sc2_features.FeatureType.CATEGORICAL:
                self.conv_layers[feature_spec.name] = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')

    def call(self, features):
        transposed = tf.transpose(features, perm=[0, 2, 3, 1])
        embed_list = []
        for index, feature_spec in enumerate(self.feature_specs):
            tensor = transposed[:, :, :, index]
            if feature_spec.type == sc2_features.FeatureType.CATEGORICAL:
                one_hot = tf.one_hot(tensor, depth=feature_spec.scale)
                embed = self.conv_layers[feature_spec.name](one_hot)
                embed_list.append(embed)
            else:   # Scalar features (최소값은 모든 feature 에서 0)
                rescale = tf.math.log(tf.cast(tensor, tf.float32) + 1.)   # log(0) = inf 이므로 전체에 1을 더해줌
                embed_list.append(tf.expand_dims(rescale, -1))  # Categorical feature channel 값과 concat 하기 위해 dimension 맞춰줌
        return tf.concat(embed_list, axis=-1)


class NonSpatialFeaturePreProcessingLayer(layers.Layer):
    def __init__(self):
        super(__class__, self).__init__()

    def call(self, features):
        return tf.math.log(tf.cast(features, tf.float32) + 1.)  # log(0) = inf 이므로 전체에 1을 더해줌


class AtariNet(tf.keras.Model):
    def __init__(self, action_spec, available_actions, available_arguments):
        super(__class__, self).__init__()
        self.action_spec = action_spec
        self.available_actions = available_actions
        self.available_arguments = available_arguments

        self.screen_feature_preprocessing_layer = SpatialFeaturePreProcessingLayer(sc2_features.SCREEN_FEATURES)

        self.screen_feature_conv1 = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same')
        self.screen_feature_conv2 = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')

        self.flatten_layer = layers.Flatten()
        self.state_representation = layers.Dense(256, activation='relu')

        self.value = layers.Dense(1)
        self.action_policy = layers.Dense(len(self.available_actions), activation='softmax')
        self.argument_policy = dict()
        for arg_type in self.available_arguments:
            if arg_type.name in ['screen', 'minimap', 'screen2']:
                self.argument_policy[arg_type.name + 'x'] = layers.Dense(arg_type.sizes[0], activation='softmax', name=arg_type.name + 'x')
                self.argument_policy[arg_type.name + 'y'] = layers.Dense(arg_type.sizes[1], activation='softmax', name=arg_type.name + 'y')
            else:
                self.argument_policy[arg_type.name] = layers.Dense(arg_type.sizes[0], activation='softmax', name=arg_type.name)

    def call(self, screen_feature):
        screen_feature = self.screen_feature_preprocessing_layer(screen_feature)

        screen_feature = self.screen_feature_conv1(screen_feature)
        screen_feature = self.screen_feature_conv2(screen_feature)

        screen_feature = self.flatten_layer(screen_feature)
        state = self.state_representation(screen_feature)

        value = self.value(state)
        action_policy = self.action_policy(state)
        argument_policy = dict()
        for arg_type in self.available_arguments:
            if arg_type.name in ['screen', 'minimap', 'screen2']:
                argument_policy[arg_type.name + 'x'] = self.argument_policy[arg_type.name + 'x'](state)
                argument_policy[arg_type.name + 'y'] = self.argument_policy[arg_type.name + 'y'](state)
            else:
                argument_policy[arg_type.name] = self.argument_policy[arg_type.name](state)
        return value, action_policy, argument_policy


class TestAgent(object):

    def __init__(self, train=True, discount_factor=0.95, learning_rate=1e-5, step_size=40):
        self.is_training = train
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        self.step_size = step_size
        self.memory = []
        self.obs_spec = None
        self.action_spec = None
        self.model = None
        self.available_actions = None
        self.available_arguments = None
        self.last_state = None
        self.last_action = None

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.available_actions = [
            FUNCTIONS.no_op,
            FUNCTIONS.select_rect,
            FUNCTIONS.Move_screen,
        ]
        self.available_arguments = [
            self.action_spec.types[arg_type.id]
            for arg_type in set(sum([action.args for action in self.available_actions], []))
        ]
        self.model = AtariNet(action_spec=self.action_spec,
                              available_actions=self.available_actions,
                              available_arguments=self.available_arguments)

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.memory = []

    def step(self, obs):
        state = self.get_state(obs)
        if self.is_training and (obs.step_type != sc2_env.StepType.FIRST):
            self.train(self.last_state, self.last_action, obs.reward, state, obs.step_type == sc2_env.StepType.LAST)
        action_id, args = self.get_action(obs, state)
        self.last_state = state
        self.last_action = (action_id, args)
        function_id = self.available_actions[action_id].id
        return sc2_actions.FunctionCall(function_id, args)

    def get_state(self, obs):
        screen_feature = obs.observation.feature_screen
        return screen_feature

    def predict(self, state):
        screen_feature = state
        return self.model(np.array([screen_feature]))

    def get_action(self, obs, state):
        value, action_policy, argument_policy = self.predict(state)
        action_policy = action_policy[0].numpy()
        action_mask = [1. if action.id in obs.observation.available_actions else 0. for action in self.available_actions]
        action_policy *= action_mask
        action_policy /= np.sum(action_policy)

        action_id = np.random.choice(range(len(action_policy)), 1, p=action_policy)[0]
        action = self.available_actions[action_id]
        args = []
        for arg_type in action.args:
            if arg_type.name in ['screen', 'minimap', 'screen2']:
                x_policy = argument_policy[arg_type.name + 'x'][0].numpy()
                y_policy = argument_policy[arg_type.name + 'y'][0].numpy()
                x = np.random.choice(range(len(x_policy)), 1, p=x_policy)[0]
                y = np.random.choice(range(len(y_policy)), 1, p=y_policy)[0]
                args.append([x, y])
            else:
                arg_policy = argument_policy[arg_type.name][0].numpy()
                arg = np.random.choice(range(len(arg_policy)), 1, p=arg_policy)[0]
                args.append([arg])
        return action_id, args

    def store(self, state, action, reward):
        screen_feature = state
        action_id, args = action
        action_one_hot = np.zeros(len(self.available_actions), dtype='float32')
        action_one_hot[action_id] = 1.
        argument_one_hot = dict()
        for arg_type in self.available_arguments:
            if arg_type.name in ['screen', 'minimap', 'screen2']:
                argument_one_hot[arg_type.name + 'x'] = np.ones(arg_type.sizes[0])
                argument_one_hot[arg_type.name + 'y'] = np.ones(arg_type.sizes[1])
            else:
                argument_one_hot[arg_type.name] = np.ones(arg_type.sizes[0])
        for argument, arg_type in zip(args, self.available_actions[action_id].args):
            if arg_type.name in ['screen', 'minimap', 'screen2']:
                argument_one_hot[arg_type.name + 'x'] = np.zeros_like(argument_one_hot[arg_type.name + 'x'])
                argument_one_hot[arg_type.name + 'y'] = np.zeros_like(argument_one_hot[arg_type.name + 'y'])
                argument_one_hot[arg_type.name + 'x'][argument[0]] = 1
                argument_one_hot[arg_type.name + 'y'][argument[1]] = 1
            else:
                argument_one_hot[arg_type.name] = np.zeros_like(argument_one_hot[arg_type.name])
                argument_one_hot[arg_type.name][argument] = 1
        self.memory.append((screen_feature, action_one_hot, argument_one_hot, reward))

    def calculate_G(self, rewards, next_state):
        G = np.zeros_like(rewards, dtype='float32')
        next_value = 0
        if next_state is not None:
            next_value, _, _ = self.predict(next_state)
            next_value = next_value[0].numpy()
        for t in reversed(range(0, len(rewards))):
            value = rewards[t] + self.discount_factor * next_value
            G[t] = value
            next_value = value
        return G

    def train(self, state, action, reward, next_state, done):
        self.store(state, action, reward)
        if done or (self.step_size and (len(self.memory) >= self.step_size)):
            screen_features = np.array([item[0] for item in self.memory])
            action_one_hots = np.array([item[1] for item in self.memory])
            argument_one_hots = dict()
            for arg_type in self.available_arguments:
                if arg_type.name in ['screen', 'minimap', 'screen2']:
                    argument_one_hots[arg_type.name + 'x'] = np.array([item[2][arg_type.name + 'x'] for item in self.memory])
                    argument_one_hots[arg_type.name + 'y'] = np.array([item[2][arg_type.name + 'y'] for item in self.memory])
                else:
                    argument_one_hots[arg_type.name] = np.array([item[2][arg_type.name] for item in self.memory])
            rewards = np.array([item[3] for item in self.memory])
            targets = self.calculate_G(rewards, None if done else next_state)
            self.memory = []
        else:
            return

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            values, action_policy, argument_policy = self.model(screen_features)
            action_probs = tf.reduce_sum(action_one_hots * action_policy, axis=1, keepdims=True)
            argument_probs = 1.
            for arg_name in argument_one_hots:
                arg_probs = tf.reduce_sum(argument_one_hots[arg_name] * argument_policy[arg_name], axis=1, keepdims=True)
                argument_probs *= arg_probs

            advantage = targets - values
            policy_loss = tf.stop_gradient(advantage) * -tf.math.log(action_probs * argument_probs + 1e-10)
            value_loss = tf.square(advantage)
            entropy_regularisation = tf.reduce_sum(action_policy * tf.math.log(action_policy + 1e-10), axis=1)
            loss = tf.reduce_sum(policy_loss + 0.5 * value_loss + 0.01 * entropy_regularisation)

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
