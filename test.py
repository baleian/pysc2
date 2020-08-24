import importlib

from absl import app
from absl import flags

from pysc2.env import run_loop
from pysc2.env import sc2_env


flags.DEFINE_string('agent', 'agents.test_agent.TestAgent',
                    'Which agent to run, as a python path to an Agent class.')

FLAGS = flags.FLAGS


def _get_cls(class_path):
    module, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module), class_name)


def main(unused_argv):
    agent_cls = _get_cls(FLAGS.agent)

    with sc2_env.SC2Env(
            map_name='CollectMineralShards',
            players=[sc2_env.Agent(sc2_env.Race['terran'])],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
            ),
            visualize=True,
            realtime=False,
    ) as env:
        run_loop.run_loop([agent_cls()], env)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
