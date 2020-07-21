from pysc2.env import sc2_env

from .base_env import BaseEnv


class MoveToBeacon(BaseEnv):

    def __init__(self, **kwargs):
        super(__class__, self).__init__(
            map_name='MoveToBeacon',
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
            ),
            **kwargs
        )
