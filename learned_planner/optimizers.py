import abc
import dataclasses
import math
from typing import Any

import torch as th
from farconf import to_dict
from stable_baselines3.common.type_aliases import check_cast
from typing_extensions import Self


@dataclasses.dataclass
class BaseLRSchedule(abc.ABC):
    @abc.abstractmethod
    def __call__(self, progress_remaining: float) -> float:
        ...

    @abc.abstractmethod
    def __mul__(self: Self, other: float) -> Self:
        ...


@dataclasses.dataclass
class FlatLRSchedule(BaseLRSchedule):
    lr: float = 0.0003

    def __call__(self, progress_remaining: float) -> float:
        return self.lr

    def __mul__(self: Self, other: float) -> Self:
        return dataclasses.replace(self, lr=self.lr * other)


@dataclasses.dataclass
class PolynomialLRSchedule(BaseLRSchedule):
    lr: float = 0.0003
    power: float = 1.0
    baseline: float = 0.0

    def __call__(self, progress_remaining: float) -> float:
        baseline = getattr(self, "baseline", 0.0)  # workaround to be able to load previous version
        return baseline + (self.lr - baseline) * (max(0.0, progress_remaining) ** self.power)

    def __mul__(self: Self, other: float) -> Self:
        return dataclasses.replace(self, lr=self.lr * other, baseline=self.baseline * other)


@dataclasses.dataclass
class InitiallyZeroLRSchedule(BaseLRSchedule):
    zero_proportion: float = 0.0001
    base: BaseLRSchedule = dataclasses.field(default_factory=FlatLRSchedule)

    def __call__(self, progress_remaining: float) -> float:
        # Progress_remaining starts at 1 and goes down -- so while it's larger than 1-zero_proportion, return 0 learning rate,
        # afterwards return a positive value.
        threshold = 1 - self.zero_proportion
        if progress_remaining > threshold:
            return 0.0  # Do not learn in the zero_proportion phase, just warm the optimizer up
        return self.base(progress_remaining / threshold)

    def __mul__(self: Self, other: float) -> Self:
        return dataclasses.replace(self, base=self.base * other)


@dataclasses.dataclass
class BaseOptimizerConfig(abc.ABC):
    lr: BaseLRSchedule = dataclasses.field(default_factory=FlatLRSchedule)

    @abc.abstractmethod
    def policy_kwargs(self) -> dict[str, Any]:
        ...


@dataclasses.dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    eps: float = 1e-8
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    amsgrad: bool = False
    foreach: bool | None = None  # Use the `foreach` implementation (faster)
    fused: bool | None = None  # Use the `fused` implementation (fastest, experimental)

    def policy_kwargs(self) -> dict[str, Any]:
        optimizer_kwargs = check_cast(dict, to_dict(self))
        del optimizer_kwargs["lr"]
        del optimizer_kwargs["_type_"]
        return dict(optimizer_class=th.optim.Adam, optimizer_kwargs=optimizer_kwargs)

    def scale_batch_size(self, from_bs: float = 1.0, to_bs: float = 0.1):
        ratio = to_bs / from_bs
        return dataclasses.replace(self, lr=self.lr * math.sqrt(ratio), betas=(self.betas[0] ** ratio, self.betas[1] ** ratio))
