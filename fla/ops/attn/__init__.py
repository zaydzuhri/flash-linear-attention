# -*- coding: utf-8 -*-

from .parallel import parallel_attn
from .naive import naive_attn

__all__ = [
    'parallel_attn',
    'naive_attn'
]
