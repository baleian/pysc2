import numpy

from pysc2.lib import actions
from pysc2.lib import features

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
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            if not beacon:
                return FUNCTIONS.no_op()
            beacon_center = numpy.mean(beacon, axis=0).round()
            return FUNCTIONS.Move_screen('now', beacon_center)
        else:
            return FUNCTIONS.select_army('select')
