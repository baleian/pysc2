import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features as sc2_features


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
    def __init__(self, action_spec, available_actions):
        super(__class__, self).__init__()
        self.action_spec = action_spec
        self.available_actions = available_actions
        self.available_arguments = [self.action_spec.types[arg_type.id] for arg_type in set(sum([action.args for action in self.available_actions], []))]

        self.screen_feature_preprocessing_layer = SpatialFeaturePreProcessingLayer(sc2_features.SCREEN_FEATURES)
        self.minimap_feature_preprocessing_layer = SpatialFeaturePreProcessingLayer(sc2_features.MINIMAP_FEATURES)
        self.player_feature_preprocessing_layer = NonSpatialFeaturePreProcessingLayer()

        self.screen_feature_conv1 = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same')
        self.screen_feature_conv2 = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.minimap_feature_conv1 = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same')
        self.minimap_feature_conv2 = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.player_feature_dense = layers.Dense(64, activation='tanh')

        self.flatten_layer = layers.Flatten()
        self.concat_layer = layers.Concatenate(axis=1)
        self.state_representation = layers.Dense(256, activation='relu')

        self.value = layers.Dense(1)
        self.action_policy = layers.Dense(len(self.available_actions), activation='softmax')
        self.argument_policy = dict()
        for arg_type in self.available_arguments:
            if arg_type.name in ['screen', 'minimap', 'screen2']:
                self.argument_policy[arg_type.name + 'x'] = layers.Dense(arg_type.sizes[0], activation='softmax')
                self.argument_policy[arg_type.name + 'y'] = layers.Dense(arg_type.sizes[1], activation='softmax')
            else:
                self.argument_policy[arg_type.name] = layers.Dense(arg_type.sizes[0], activation='softmax')

    def call(self, screen_feature, minimap_feature, player_feature):
        screen_feature = self.screen_feature_preprocessing_layer(screen_feature)
        minimap_feature = self.minimap_feature_preprocessing_layer(minimap_feature)
        player_feature = self.player_feature_preprocessing_layer(player_feature)

        screen_feature = self.screen_feature_conv1(screen_feature)
        screen_feature = self.screen_feature_conv2(screen_feature)
        minimap_feature = self.minimap_feature_conv1(minimap_feature)
        minimap_feature = self.minimap_feature_conv2(minimap_feature)
        player_feature = self.player_feature_dense(player_feature)

        screen_feature = self.flatten_layer(screen_feature)
        minimap_feature = self.flatten_layer(minimap_feature)
        player_feature = self.flatten_layer(player_feature)

        concat = self.concat_layer([screen_feature, minimap_feature, player_feature])
        state = self.state_representation(concat)

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

    def __init__(self):
        self.obs_spec = None
        self.action_spec = None
        self.model = None
        self.available_actions = [
            FUNCTIONS.no_op,
            FUNCTIONS.select_rect,
            FUNCTIONS.Move_screen,
        ]

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.model = AtariNet(action_spec=self.action_spec, available_actions=self.available_actions)

    def reset(self):
        pass

    def step(self, obs):
        screen_feature = np.array([obs.observation.feature_screen])
        minimap_feature = np.array([obs.observation.feature_minimap])
        player_feature = np.array([obs.observation.player])
        value, action_policy, argument_policy = self.model(screen_feature, minimap_feature, player_feature)

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
        return sc2_actions.FunctionCall(action.id, args)
