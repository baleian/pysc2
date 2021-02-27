import numpy

import sys
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import actions
from .base_agent import BaseAgent

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS


def _xy_locs(mask):
    y, x = mask.nonzero()
    return list(zip(x, y))


class MoveToBeaconSimpleAgent(BaseAgent):

    def step(self, obs):
        super(__class__, self).step(obs)

        print(self.action_spec.functions)

        for function_id in range(573):
            print('function_id:', function_id)
            print([[size for size in arg.sizes] for arg in self.action_spec.functions[function_id].args])

        sys.exit(0)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            if not beacon:
                return FUNCTIONS.no_op()
            beacon_center = numpy.mean(beacon, axis=0).round()
            return FUNCTIONS.Move_screen([0], beacon_center)
        else:
            return FUNCTIONS.select_army([0])


class MoveToBeaconDQNAgent(BaseAgent):

    def step(self, obs):
        super(__class__, self).step(obs)

