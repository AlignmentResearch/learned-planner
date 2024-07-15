import abc
import dataclasses

from torch import nn


@dataclasses.dataclass
class ActivationFnConfig(abc.ABC):
    @abc.abstractproperty
    def fn(self) -> type[nn.Module]:
        ...

    @abc.abstractproperty
    def name(self) -> str:
        ...


@dataclasses.dataclass
class IdentityActConfig(ActivationFnConfig):
    @property
    def fn(self):
        return nn.Identity

    @property
    def name(self) -> str:
        return "identity"


@dataclasses.dataclass
class TanhConfig(ActivationFnConfig):
    @property
    def fn(self):
        return nn.Tanh

    @property
    def name(self) -> str:
        return "tanh"


@dataclasses.dataclass
class ReLUConfig(ActivationFnConfig):
    @property
    def fn(self):
        return nn.ReLU

    @property
    def name(self) -> str:
        return "relu"
