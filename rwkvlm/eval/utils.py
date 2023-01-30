from collections import OrderedDict

from torch import Tensor

__all__ = ['prefixed_params']


def prefixed_params(params: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {
        k[len(prefix):]: t
        for k, t in params.items()
        if k.startswith(prefix)
    }
