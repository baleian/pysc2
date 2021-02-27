import importlib

from absl import app
from absl import flags

from pysc2.env import run_loop

flags.DEFINE_string('env', None, 'Environment to use.')
flags.DEFINE_string('agent', None, 'Which agent to train, as a python path to an Agent class.')
flags.DEFINE_string('model_dir', None, 'Directory path where the model will be saved.')

flags.mark_flag_as_required('env')
flags.mark_flag_as_required('agent')

FLAGS = flags.FLAGS


def _get_cls(class_path):
    module, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module), class_name)


def main(unused_argv):
    env_cls = _get_cls(FLAGS.env)
    agent_cls = _get_cls(FLAGS.agent)

    env = env_cls(
        visualize=True,
        realtime=True,
    )

    run_loop.run_loop([agent_cls()], env)

    env.close()


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
