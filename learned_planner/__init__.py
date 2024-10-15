import os
import pathlib

LP_DIR = pathlib.Path(__file__).parent.parent
ON_CLUSTER = os.path.exists("/training")
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000/"  # DRC(3, 3) 2B checkpoint
if ON_CLUSTER:
    BOXOBAN_CACHE = pathlib.Path("/training/.sokoban_cache/")
else:
    BOXOBAN_CACHE = LP_DIR / "training/.sokoban_cache/"
