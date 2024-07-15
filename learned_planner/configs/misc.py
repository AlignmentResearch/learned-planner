import random
from pathlib import Path


def random_seed() -> int:
    return random.randint(0, 2**31 - 1)


DEFAULT_TRAINING: Path = Path(__file__).parent.parent / "training"
