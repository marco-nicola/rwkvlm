import torch
from torch import Tensor
from torch.nn import Identity, Module

from .channel_mix import ChannelMix
from .layer_norm import LayerNorm
from .state import State
from .time_mix import TimeMix
from .utils import prefixed_params

__all__ = ['Block']


class Block(Module):
    __constants__ = ['must_rescale']
    # time_mix: TimeMix
    # chan_mix: ChannelMix
    # ln0: LayerNorm | Identity
    # ln1: LayerNorm
    # ln2: LayerNorm
    must_rescale: bool

    def __init__(
            self,
            params: dict[str, Tensor],
            block_num: int,
            emb_size: int,
            dtype: torch.dtype,
            device: torch.device | str,
            rescale_layer: int = 6,  # set x=x/2 every X layer
    ) -> None:
        def convert(t: Tensor) -> Tensor:
            return t.to(dtype=dtype, device=device)

        super().__init__()

        self.time_mix = TimeMix(
            prefixed_params(params, 'att.'),
            block_num=block_num,
            rescale_layer=rescale_layer,
            dtype=dtype,
            device=device
        )
        self.chan_mix = ChannelMix(
            prefixed_params(params, 'ffn.'),
            block_num=block_num,
            rescale_layer=rescale_layer,
            dtype=dtype,
            device=device
        )
        self.ln0 = LayerNorm(
            (emb_size,),
            convert(params['ln0.weight']),
            convert(params['ln0.bias']),
        ) if 'ln0.weight' in params else Identity()
        self.ln1 = LayerNorm(
            (emb_size,),
            convert(params['ln1.weight']),
            convert(params['ln1.bias']),
        )
        self.ln2 = LayerNorm(
            (emb_size,),
            convert(params['ln2.weight']),
            convert(params['ln2.bias']),
        )
        self.must_rescale = ((block_num + 1) % rescale_layer == 0)

    def forward(self, x: Tensor, state: State) -> Tensor:
        x = self.ln0(x)
        x += self.time_mix(self.ln1(x), state)
        x += self.chan_mix(self.ln2(x), state)
        if self.must_rescale:
            x /= 2
        return x
