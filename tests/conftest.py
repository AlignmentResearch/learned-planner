import warnings
from pathlib import Path

import pytest

with warnings.catch_warnings():
    # Import tensorboard here first so we can suppress DeprecationWarnings while testing.
    # warning would be fixed in https://github.com/pytorch/pytorch/pull/88524
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message="distutils Version classes are", module="torch.utils.tensorboard"
    )
    import torch.utils.tensorboard  # noqa:F401  # type: ignore


@pytest.fixture
def BOXOBAN_CACHE():
    return Path(__file__).parent
