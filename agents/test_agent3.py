import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features as sc2_features
from pysc2.env import environment as sc2_env


AVAILABLE_ACTIONS = [
    # sc2_actions.FUNCTIONS.no_op,
    # sc2_actions.FUNCTIONS.select_rect,
    sc2_actions.FUNCTIONS.Move_screen,
]

AVAILABLE_ARGUMENTS = [
    # sc2_actions.TYPES.queued,
    # sc2_actions.TYPES.select_add,
    sc2_actions.TYPES.screen,
    # sc2_actions.TYPES.screen2,
]

DEFAULT_ARGUMENTS = {
    # sc2_actions.select_add: [0],
    sc2_actions.TYPES.queued: [0],
}

AVAILABLE_SCREEN_FEATURES = [
    sc2_features.SCREEN_FEATURES.player_relative,
    # sc2_features.SCREEN_FEATURES.selected,
]

AVAILABLE_MINIMAP_FEATURES = [
    # sc2_features.MINIMAP_FEATURES.visibility_map,
    # sc2_features.MINIMAP_FEATURES.player_relative,
    # sc2_features.MINIMAP_FEATURES.selected,
]

AVAILABLE_PLAYER_FEATURES = [
    # sc2_features.Player.army_count,
]


class SpatialFeaturePreProcessingLayer(layers.Layer):
    def __init__(self, available_features):
        super(__class__, self).__init__()
        self.available_features = available_features
        self.conv_layers = dict()
        for feature in self.available_features:
            if feature.type == sc2_features.FeatureType.CATEGORICAL:
                self.conv_layers[feature.name] = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')

    def call(self, features):
        transposed = tf.transpose(features, perm=[0, 2, 3, 1])
        embed_list = []
        for index, feature in enumerate(self.available_features):
            tensor = transposed[:, :, :, index]
            if feature.type == sc2_features.FeatureType.CATEGORICAL:
                one_hot = tf.one_hot(tensor, depth=feature.scale)
                embed = self.conv_layers[feature.name](one_hot)
                embed_list.append(embed)
            else:   # Scalar features (최소값은 모든 feature 에서 0)
                rescale = tf.math.log(tf.cast(tensor, tf.float32) + 1.)   # log(0) = inf 이므로 전체에 1을 더해줌
                embed_list.append(tf.expand_dims(rescale, -1))  # Categorical feature channel 값과 concat 하기 위해 dimension 맞춰줌
        return tf.concat(embed_list, axis=-1)


class NonSpatialFeaturePreProcessingLayer(layers.Layer):
    def __init__(self, available_features):
        super(__class__, self).__init__()
        self.available_features = available_features

    def call(self, features):
        return tf.math.log(tf.cast(features, tf.float32) + 1.)  # log(0) = inf 이므로 전체에 1을 더해줌


class AtariNet(tf.keras.Model):
    def __init__(self,
                 available_screen_features,
                 available_minimap_features,
                 available_player_features,
                 available_actions,
                 available_arguments):
        super(__class__, self).__init__()
        self.available_screen_features = available_screen_features
        self.available_minimap_features = available_minimap_features
        self.available_player_features = available_player_features
        self.available_actions = available_actions
        self.available_arguments = available_arguments

        self.screen_feature_preprocessing_layer = SpatialFeaturePreProcessingLayer(self.available_screen_features)
        self.minimap_feature_preprocessing_layer = SpatialFeaturePreProcessingLayer(self.available_minimap_features)
        self.player_feature_preprocessing_layer = NonSpatialFeaturePreProcessingLayer(self.available_player_features)

        self.screen_feature_conv1 = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same')   # TODO: Activation
        self.screen_feature_conv2 = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.minimap_feature_conv1 = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same')
        self.minimap_feature_conv2 = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.player_feature_dense = layers.Dense(64, activation='tanh')

        self.flatten_layer = layers.Flatten()
        self.concat_layer = layers.Concatenate(axis=1)
        self.state_representation = layers.Dense(256, activation='relu')

        self.value = layers.Dense(1)
        self.action_policy = layers.Dense(len(self.available_actions), activation='softmax')
        self.arguments_policy = dict()
        for argument_type in self.available_arguments:
            if len(argument_type.sizes) == 2:    # Spatial action policy
                self.arguments_policy[argument_type.name + 'x'] = layers.Dense(argument_type.sizes[0], activation='softmax', name=argument_type.name + 'x')
                self.arguments_policy[argument_type.name + 'y'] = layers.Dense(argument_type.sizes[1], activation='softmax', name=argument_type.name + 'y')
            else:   # Non-spatial action policy
                self.arguments_policy[argument_type.name] = layers.Dense(argument_type.sizes[0], activation='softmax', name=argument_type.name)

    def call(self, screen_feature, minimap_feature, player_feature):
        concat_layer_inputs = []

        if len(self.available_screen_features) > 0:
            screen_feature = self.screen_feature_preprocessing_layer(screen_feature)
            screen_feature = self.screen_feature_conv1(screen_feature)
            screen_feature = self.screen_feature_conv2(screen_feature)
            screen_feature = self.flatten_layer(screen_feature)
            concat_layer_inputs.append(screen_feature)

        if len(self.available_minimap_features) > 0:
            minimap_feature = self.minimap_feature_preprocessing_layer(minimap_feature)
            minimap_feature = self.minimap_feature_conv1(minimap_feature)
            minimap_feature = self.minimap_feature_conv2(minimap_feature)
            minimap_feature = self.flatten_layer(minimap_feature)
            concat_layer_inputs.append(minimap_feature)

        if len(self.available_player_features) > 0:
            player_feature = self.player_feature_preprocessing_layer(player_feature)
            player_feature = self.player_feature_dense(player_feature)
            concat_layer_inputs.append(player_feature)

        if len(concat_layer_inputs) > 1:
            concat = self.concat_layer(concat_layer_inputs)
        else:
            concat = concat_layer_inputs[0]

        state = self.state_representation(concat)
        value = self.value(state)
        action_policy = self.action_policy(state)
        arguments_policy = dict()
        for argument_type in self.available_arguments:
            if len(argument_type.sizes) == 2:
                arguments_policy[argument_type.name + 'x'] = self.arguments_policy[argument_type.name + 'x'](state)
                arguments_policy[argument_type.name + 'y'] = self.arguments_policy[argument_type.name + 'y'](state)
            else:
                arguments_policy[argument_type.name] = self.arguments_policy[argument_type.name](state)
        return value, action_policy, arguments_policy


class TestAgent(object):

    def __init__(self, train=True, discount_factor=0.99, learning_rate=0.0001, step_size=20):
        self.available_screen_features = AVAILABLE_SCREEN_FEATURES
        self.available_minimap_features = AVAILABLE_MINIMAP_FEATURES
        self.available_player_features = AVAILABLE_PLAYER_FEATURES
        self.available_actions = AVAILABLE_ACTIONS
        self.available_arguments = AVAILABLE_ARGUMENTS
        self.available_argument_ids = [argument.id for argument in AVAILABLE_ARGUMENTS]
        self.default_arguments = DEFAULT_ARGUMENTS

        self.is_training = train
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.optimizer = optimizers.RMSprop(learning_rate=learning_rate)
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
        # Dynamically set arguments spec according to the environment
        self.available_arguments = [self.action_spec.types[argument_type.id] for argument_type in self.available_arguments]
        self.model = AtariNet(available_screen_features=self.available_screen_features,
                              available_minimap_features=self.available_minimap_features,
                              available_player_features=self.available_player_features,
                              available_actions=self.available_actions,
                              available_arguments=self.available_arguments)

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.memory = []

    def step(self, obs):
        if obs.step_type == sc2_env.StepType.FIRST:
            return sc2_actions.FUNCTIONS.select_army([0])

        state = self.get_state(obs)
        reward = obs.reward
        action_id, arguments = self.get_action(obs, state)
        done = obs.step_type == sc2_env.StepType.LAST

        if self.is_training and self.last_state and self.last_action:
            self.train(self.last_state, self.last_action, reward, state, done)
        self.last_state = state
        self.last_action = (action_id, arguments)

        function_id = AVAILABLE_ACTIONS[action_id].id
        return sc2_actions.FunctionCall(function_id, arguments)

    def get_state(self, obs):
        screen_feature = [obs.observation.feature_screen[feature.name] for feature in self.available_screen_features]
        minimap_feature = [obs.observation.feature_minimap[feature.name] for feature in self.available_minimap_features]
        player_feature = [obs.observation.player[feature.name] for feature in self.available_player_features]
        return screen_feature, minimap_feature, player_feature

    def predict(self, state):
        screen_feature, minimap_feature, player_feature = state
        return self.model(np.array([screen_feature]), np.array([minimap_feature]), np.array([player_feature]))

    def get_action(self, obs, state):
        value, action_policy, arguments_policy = self.predict(state)
        action_policy = action_policy[0].numpy()

        # 현재 환경 상태에 적용 가능한 액션들만 취하도록 masking 및 확률 재분배
        action_mask = [1. if action.id in obs.observation.available_actions else 0. for action in self.available_actions]
        action_policy *= action_mask
        action_policy /= np.sum(action_policy)

        action_id = np.random.choice(range(len(action_policy)), 1, p=action_policy)[0]
        arguments = []
        for argument_type in self.available_actions[action_id].args:
            if argument_type.id not in self.available_argument_ids:
                arguments.append(self.default_arguments[argument_type])
            elif len(argument_type.sizes) == 2:
                x_policy = arguments_policy[argument_type.name + 'x'][0].numpy()
                y_policy = arguments_policy[argument_type.name + 'y'][0].numpy()
                x = np.random.choice(range(len(x_policy)), 1, p=x_policy)[0]
                y = np.random.choice(range(len(y_policy)), 1, p=y_policy)[0]
                arguments.append([x, y])
            else:
                arg_policy = arguments_policy[argument_type.name][0].numpy()
                arg = np.random.choice(range(len(arg_policy)), 1, p=arg_policy)[0]
                arguments.append([arg])
        return action_id, arguments

    def store(self, state, action, reward):
        screen_feature, minimap_feature, player_feature = state
        action_id, arguments = action
        action_one_hot = np.zeros(len(self.available_actions), dtype='float32')
        action_one_hot[action_id] = 1.
        arguments_one_hot = dict()
        for argument_type in self.available_arguments:
            if len(argument_type.sizes) == 2:
                arguments_one_hot[argument_type.name + 'x'] = np.ones(argument_type.sizes[0])
                arguments_one_hot[argument_type.name + 'y'] = np.ones(argument_type.sizes[1])
            else:
                arguments_one_hot[argument_type.name] = np.ones(argument_type.sizes[0])
        for argument, argument_type in zip(arguments, self.available_actions[action_id].args):
            if argument_type.id not in self.available_argument_ids:
                continue
            if len(argument_type.sizes) == 2:
                arguments_one_hot[argument_type.name + 'x'] = np.zeros_like(arguments_one_hot[argument_type.name + 'x'])
                arguments_one_hot[argument_type.name + 'y'] = np.zeros_like(arguments_one_hot[argument_type.name + 'y'])
                arguments_one_hot[argument_type.name + 'x'][argument[0]] = 1
                arguments_one_hot[argument_type.name + 'y'][argument[1]] = 1
            else:
                arguments_one_hot[argument_type.name] = np.zeros_like(arguments_one_hot[argument_type.name])
                arguments_one_hot[argument_type.name][argument] = 1
        self.memory.append((screen_feature, minimap_feature, player_feature, action_one_hot, arguments_one_hot, reward))

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
            minimap_features = np.array([item[1] for item in self.memory])
            player_features = np.array([item[2] for item in self.memory])
            action_one_hots = np.array([item[3] for item in self.memory])
            arguments_one_hots = dict()
            for argument_type in self.available_arguments:
                if len(argument_type.sizes) == 2:
                    arguments_one_hots[argument_type.name + 'x'] = np.array([item[4][argument_type.name + 'x'] for item in self.memory])
                    arguments_one_hots[argument_type.name + 'y'] = np.array([item[4][argument_type.name + 'y'] for item in self.memory])
                else:
                    arguments_one_hots[argument_type.name] = np.array([item[4][argument_type.name] for item in self.memory])
            rewards = np.array([item[5] for item in self.memory])
            targets = self.calculate_G(rewards, None if done else next_state)
            self.memory = []
        else:
            return

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            values, action_policy, arguments_policy = self.model(screen_features, minimap_features, player_features)
            targets = tf.convert_to_tensor(targets[:, None], dtype=tf.float32)

            action_probs = tf.reduce_sum(action_one_hots * action_policy, axis=1, keepdims=True)
            cross_entropy = 0
            for argument_name in arguments_one_hots:
                argument_one_hots = arguments_one_hots[argument_name]
                argument_policy = arguments_policy[argument_name]
                argument_probs = tf.reduce_sum(argument_one_hots * argument_policy, axis=1, keepdims=True)
                cross_entropy += -tf.math.log(action_probs * argument_probs)

            advantages = targets - values

            policy_loss = tf.reduce_mean(tf.stop_gradient(advantages) * cross_entropy)
            value_loss = 0.5 * tf.reduce_mean(tf.square(advantages))
            # entropy_regularisation = 0.01 * entropy

            loss = policy_loss + value_loss     # + tf.stop_gradient(entropy_regularisation)

            # entropy_regularisation = tf.reduce_sum(action_policy * tf.math.log(action_policy + 1e-10), axis=1, keepdims=True)
            # loss = tf.reduce_mean(policy_loss) + 0.5 * tf.reduce_mean(value_loss) # + 0.01 * tf.reduce_sum(entropy_regularisation)

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
