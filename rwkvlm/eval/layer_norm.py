from typing import Tuple

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import layer_norm

__all__ = ['LayerNorm']


class LayerNorm(Module):
    __constants__ = ['normalized_shape']
    normalized_shape: Tuple[int, ...]
    weight: Tensor
    bias: Tensor

    def __init__(
            self,
            normalized_shape: Tuple[int],
            weight: Tensor,
            bias: Tensor,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = weight
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.normalized_shape, self.weight, self.bias)
