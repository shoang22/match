"""Training configs as scripts/match/train_match.py parses them, for the config tests."""


def parsed_options(config, bindings=None):
  """Options as scripts/match/train_match.py parses them from `config` (and gin `bindings`)."""
  import gin
  from match.options import Options
  from match.utils import gin_util  # noqa: F401 (registers join_strings)
  gin.clear_config()
  try:
    gin.parse_config_files_and_bindings([config], bindings=bindings, skip_unknown=True)
    return Options()
  finally:
    gin.clear_config()
