import argparse
import os
import pathlib

LP_DIR = pathlib.Path(__file__).parent.parent

if "editable" in str(LP_DIR):
    LP_DIR = LP_DIR.parent.parent

ON_CLUSTER = os.path.exists("/training")
DRC33_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000/"  # DRC(3, 3) 2B checkpoint
DRC11_PATH_IN_REPO = "drc11/eue6pax7/cp_2002944000"  # DRC(1, 1) 2B checkpoint
RESNET_PATH_IN_REPO = "resnet/syb50iz7/cp_2002944000"  # ResNet(9) 2B checkpoint

MODEL_PATH_IN_REPO = DRC33_PATH_IN_REPO  # Default to DRC(3, 3) unless overridden

if ON_CLUSTER:
    BOXOBAN_CACHE = pathlib.Path("/training/sokoban_cache/")
    if not BOXOBAN_CACHE.exists():
        BOXOBAN_CACHE = pathlib.Path("/opt/sokoban_cache/")
        if not BOXOBAN_CACHE.exists():
            raise FileNotFoundError("No Sokoban cache found at /training/.sokoban_cache/ or /opt/sokoban_cache/")
else:
    BOXOBAN_CACHE = LP_DIR / "training/.sokoban_cache/"


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        # ZMQInteractiveShell is used by Jupyter notebooks and qtconsole
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


IS_NOTEBOOK = is_notebook()


def get_default_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Creates a namespace object containing only the default values
    defined in an ArgumentParser.

    It iterates through the parser's actions, retrieves the default value
    for each destination using the public get_default() method, and populates
    a new Namespace object.

    Args:
        parser: An initialized argparse.ArgumentParser object.

    Returns:
        An argparse.Namespace object populated with the default values
        from the parser's arguments.
    """
    default_namespace = argparse.Namespace()

    for action in parser._actions:
        # We are interested in actions that store a value and have a
        # destination name. We exclude the default 'help' action and
        # actions where the destination is suppressed.
        # We also check if the action object actually has a 'default' attribute.
        if action.dest != "help" and action.dest is not argparse.SUPPRESS and hasattr(action, "default"):
            default_value = parser.get_default(action.dest)
            setattr(default_namespace, action.dest, default_value)

    return default_namespace
