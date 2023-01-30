import torch
from torch import relu, sigmoid, square, Tensor
from torch.nn import Module

from .state import State

__all__ = ['ChannelMix']


class ChannelMix(Module):
    __constants__ = ['state_index']
    state_index: int
    w_key: Tensor
    w_rec: Tensor
    w_val: Tensor
    t_mix_k: Tensor
    t_mix_r: Tensor

    def __init__(
            self,
            params: dict[str, Tensor],
            block_num: int,
            dtype: torch.dtype,
            device: torch.device | str,
            rescale_layer: int = 6,
    ) -> None:
        def convert(t: Tensor) -> Tensor:
            return t.to(dtype=dtype, device=device)

        super().__init__()

        self.state_index = 5 * block_num + 0

        out_scale = 2 ** int(block_num // rescale_layer)

        self.w_key = convert(params['key.weight'])
        self.w_rec = convert(params['receptance.weight'])
        self.w_val = convert(params['value.weight'] / out_scale)
        self.t_mix_k = convert(params['time_mix_k'].squeeze())
        self.t_mix_r = convert(params['time_mix_r'].squeeze())

    def forward(self, x: Tensor, state: State) -> Tensor:
        state_val = state[self.state_index].to(x.dtype)
        xk = x * self.t_mix_k + state_val * (1 - self.t_mix_k)
        xr = x * self.t_mix_r + state_val * (1 - self.t_mix_r)
        state[self.state_index] = x.to(state.dtype)

        r = sigmoid(self.w_rec @ xr)
        k = square(relu(self.w_key @ xk))
        kv = self.w_val @ k

        return r * kv
