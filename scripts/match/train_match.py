from match.runner import MatchRunner
import argparse
import gin
from match.options import Options
from match.utils import gin_util


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_configs", action="append", default=[])
    # applied after the configs, e.g. --gin_bindings "Options.temporal_loss_weight = 0.01" --gin_bindings "NAME = '<run>'"
    parser.add_argument("--gin_bindings", action="append", default=[])
    return parser.parse_args(argv)


def parse_options(argv: list[str] | None = None) -> Options:
    """The training Options from the command line's gin configs, overridden by its gin bindings."""
    FLAGS = parse_args(argv)
    gin.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs, bindings=FLAGS.gin_bindings, skip_unknown=True
    )
    return Options()


def main(argv: list[str]|None = None):
    # explicitly initialize the warm pool as the first line of main!
    opt = parse_options(argv)
    runner = MatchRunner(opt)
    runner.prepare_training(save_config_files=parse_args(argv).gin_configs)
    runner.run_training()
    runner.graceful_exit(0)



if __name__ == "__main__":
  main()
