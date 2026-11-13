"""scripts/match/eval_sequence.py loaded once as a module, for the tests that call its functions.

Loading it registers its gin configurables, and gin refuses to register them a second time, so every test
module gets the same module object from `load()`. It pulls in the runner (~12 s).
"""
import importlib.util
import sys

MODULE_NAME = 'eval_sequence'


def load():
    if MODULE_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, 'scripts/match/eval_sequence.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[MODULE_NAME]
            raise
    return sys.modules[MODULE_NAME]
