import torch
from torch import Tensor
from torch.nn import Sequential

from .block import Block
from .state import State
from .utils import prefixed_params

__all__ = ['Blocks']


class Blocks(Sequential):
    def __init__(
            self,
            params: dict[str, Tensor],
            emb_size: int,
            rescale_layer: int,
            dtype: torch.dtype,
            device: torch.device | str,
    ):
        super().__init__(*(
            Block(
                prefixed_params(params, f'{block_num}.'),
                block_num=block_num,
                emb_size=emb_size,
                rescale_layer=rescale_layer,
                dtype=dtype,
                device=device
            )
            for block_num in range(count_blocks(params))
        ))

    def forward(self, x: Tensor, state: State) -> Tensor:
        for module in self:
            x = module(x, state)
        return x


def count_blocks(params: dict[str, Tensor]) -> int:
    return max(int(k[:k.index('.')]) for k in params) + 1
