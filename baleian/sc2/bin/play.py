import importlib

from absl import app
from absl import flags

from pysc2.env import run_loop


flags.DEFINE_string('env', None, 'Environment to use.')
flags.DEFINE_string('agent', 'pysc2.agents.random_agent.RandomAgent',
                    'Which agent to run, as a python path to an Agent class.')
flags.DEFINE_string('model_dir', None, 'Directory path where the model to be used by the agent is stored.')

flags.mark_flag_as_required('env')

FLAGS = flags.FLAGS


def _get_cls(class_path):
    module, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module), class_name)


def main(unused_argv):
    env_cls = _get_cls(FLAGS.env)
    agent_cls = _get_cls(FLAGS.agent)

    with env_cls(
        visualize=True,
        realtime=True,
    ) as env:
        run_loop.run_loop([agent_cls()], env)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
