from absl import flags

from baleian.sc2.bin.play import entry_point

FLAGS = flags.FLAGS

FLAGS.env = 'baleian.sc2.envs.MoveToBeacon'
FLAGS.agent = 'baleian.sc2.agents.MoveToBeaconSimpleAgent'


if __name__ == "__main__":
    entry_point()
