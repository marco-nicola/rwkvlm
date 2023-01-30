import gc
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, zeros
from torch.nn import Module

from .blocks import Blocks
from .layer_norm import LayerNorm
from .state import State
from .utils import prefixed_params

__all__ = ['RWKV']


class RWKV(Module):
    __constants__ = ['_dtype', '_device', 'ctx_len', 'emb_size', 'has_pos_emb']

    _dtype: torch.dtype
    _device: Union[torch.device, str]
    ctx_len: int
    emb_weight: Tensor
    emb_size: int
    has_pos_emb: bool
    pos_emb: Optional[Tensor]
    # layer_norm: LayerNorm
    head_weight: Tensor
    # blocks: Blocks

    def __init__(
            self,
            model_name: str,
            dtype: torch.dtype,
            device: Union[torch.device, str] = 'cpu',
            rescale_layer: int = 6,  # set x=x/2 every X layer
            ctx_len: int = 1024,
    ):
        def convert(t: Tensor) -> Tensor:
            return t.to(dtype=dtype, device=device)

        super().__init__()

        with torch.no_grad():
            self._dtype = dtype
            self._device = device
            self.ctx_len = ctx_len

            params: dict[str, Tensor] = torch.load(model_name, map_location='cpu')

            self.emb_weight = convert(params['emb.weight'])
            self.emb_size = self.emb_weight.size(dim=1)
            self.head_weight = convert(params['head.weight'])

            self.layer_norm = LayerNorm(
                (self.emb_size,),
                convert(params['ln_out.weight']),
                convert(params['ln_out.bias']),
            )

            self.pos_emb = get_pos_emb(params, ctx_len)
            self.has_pos_emb = self.pos_emb is not None
            if self.has_pos_emb:
                self.pos_emb = convert(self.pos_emb)

            self.blocks = Blocks(
                params=prefixed_params(params, 'blocks.'),
                emb_size=self.emb_size,
                rescale_layer=rescale_layer,
                dtype=dtype,
                device=device,
            )

            self.eval()
            gc.collect()
            torch.cuda.empty_cache()

    @torch.jit.export
    def new_state(self) -> State:
        num_blocks = len(self.blocks)
        state = zeros(num_blocks * 5, self.emb_size, dtype=torch.float, device=self._device)
        for i in range(num_blocks):
            state[5 * i + 4] -= 1e30
        return state

    def forward(self, ctx: List[int], state: State) -> Tuple[Tensor, State]:
        with torch.no_grad():
            x, state = self._preprocess(ctx, state)
            x = self.layer_norm(x)
            x = self.head_weight @ x
            return x.float(), state

    @torch.jit.export
    def forward_preprocess(self, ctx: List[int], state: State) -> State:
        with torch.no_grad():
            return self._preprocess(ctx, state)[1]

    def _preprocess(self, ctx: List[int], state: State) -> Tuple[Tensor, State]:
        x = self.emb_weight[ctx[-1]]

        if self.has_pos_emb:
            x += self.pos_emb[len(ctx) - 1]

        x = self.blocks(x, state)

        return x, state


def get_pos_emb(params: dict[str, Tensor], ctx_len: int) -> Optional[Tensor]:
    if 'pos_emb_x' in params:
        return (params['pos_emb_x'] + params['pos_emb_y']).reshape(ctx_len + 1, -1)[:-1, :]
    return None
