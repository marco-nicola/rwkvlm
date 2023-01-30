import torch
from torch import exp, maximum, sigmoid, Tensor
from torch.nn import Module

from .state import State

__all__ = ['TimeMix']


class TimeMix(Module):
    __constants__ = ['base_state_index']
    base_state_index: int
    w_key: Tensor
    w_out: Tensor
    w_rec: Tensor
    w_val: Tensor
    t_decay: Tensor
    t_first: Tensor
    t_mix_k: Tensor
    t_mix_r: Tensor
    t_mix_v: Tensor

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

        self.base_state_index = 5 * block_num

        out_scale = 2 ** int(block_num // rescale_layer)

        self.w_key = convert(params['key.weight'])
        self.w_out = convert(params['output.weight'] / out_scale)
        self.w_rec = convert(params['receptance.weight'])
        self.w_val = convert(params['value.weight'])

        self.t_decay = -exp(params['time_decay'].squeeze().float()).to(device=device)
        self.t_first = params['time_first'].squeeze().float().to(device=device)
        self.t_mix_k = convert(params['time_mix_k'].squeeze())
        self.t_mix_r = convert(params['time_mix_r'].squeeze())
        self.t_mix_v = convert(params['time_mix_v'].squeeze())

    def forward(self, x: Tensor, state: State) -> Tensor:
        state_xx = state[self.base_state_index + 1].to(x.dtype)
        xk = x * self.t_mix_k + state_xx * (1 - self.t_mix_k)
        xv = x * self.t_mix_v + state_xx * (1 - self.t_mix_v)
        xr = x * self.t_mix_r + state_xx * (1 - self.t_mix_r)
        state[self.base_state_index + 1] = x.to(state.dtype)

        r = sigmoid(self.w_rec @ xr)
        k = self.w_key @ xk
        v = self.w_val @ xv

        kk = k.float()
        vv = v.float()

        aa = state[self.base_state_index + 2]
        bb = state[self.base_state_index + 3]
        pp = state[self.base_state_index + 4]

        ww = self.t_first + kk
        p = maximum(pp, ww)

        e1 = exp(pp - p)
        e2 = exp(ww - p)

        a = e1 * aa + e2 * vv
        b = e1 * bb + e2

        ww = pp + self.t_decay
        p = maximum(ww, kk)

        e1 = exp(ww - p)
        e2 = exp(kk - p)

        state[self.base_state_index + 2] = e1 * aa + e2 * vv
        state[self.base_state_index + 3] = e1 * bb + e2
        state[self.base_state_index + 4] = p

        wkv = (a / b).type(x.dtype)
        y: Tensor = self.w_out @ (r * wkv)
        return y
