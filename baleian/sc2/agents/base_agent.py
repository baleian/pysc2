from pysc2.agents import base_agent


class BaseAgent(base_agent.BaseAgent):

    def step(self, obs):
        super(__class__, self).step(obs)
