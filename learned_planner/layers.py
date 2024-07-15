import abc
import dataclasses

import torch


class NormConfig(abc.ABC):
    @abc.abstractmethod
    def make(self, shape: tuple[int, ...]) -> torch.nn.Module:
        ...


@dataclasses.dataclass
class IdentityNormConfig(NormConfig):
    """A 'normalization' layer that doesn't do anything."""

    def make(self, shape: tuple[int, ...]) -> torch.nn.Module:
        return torch.nn.Identity()


@dataclasses.dataclass
class LayerNormConfig(NormConfig):
    elementwise_affine: bool = True  # Have learnable parameters? y/n
    bias: bool = True  # Learnable bias? y/n
    eps: float = 1e-5

    def make(self, shape: tuple[int, ...]) -> torch.nn.Module:
        return torch.nn.LayerNorm(torch.Size(shape), eps=self.eps, elementwise_affine=self.elementwise_affine, bias=self.bias)


@dataclasses.dataclass
class RMSNormConfig(NormConfig):
    eps: float = 1e-5

    def make(self, shape: tuple[int, ...]) -> torch.nn.Module:
        return RMSNorm(torch.Size(shape), eps=self.eps)


# from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(torch.nn.Module):
    "Normalizes over the first dimension of `d` only"

    def __init__(self, d: torch.Size, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dims = max(1, len(d))

    def forward(self, x):
        output = x * torch.rsqrt(x.square().mean(-self.mean_dims, keepdim=True) + self.eps)
        return output
