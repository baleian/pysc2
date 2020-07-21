from pysc2.env import sc2_env


class BaseEnv(sc2_env.SC2Env):

    def __init__(self, **kwargs):
        super(__class__, self).__init__(**kwargs)
