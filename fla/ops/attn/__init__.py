# -*- coding: utf-8 -*-

from .parallel import parallel_attn
from .parallel_rectified import parallel_rectified_attn

__all__ = [
    'parallel_attn',
    'parallel_rectified_attn',
]
