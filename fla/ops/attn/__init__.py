# -*- coding: utf-8 -*-

from .parallel import parallel_attn
from .parallel_rectified import parallel_rectified_attn
from .parallel_softpick import parallel_softpick_attn
from .parallel_relusoftpick import parallel_relu_softpick_1_attn
from .parallel_relu_softpick_2 import parallel_relu_softpick_2_attn
from .parallel_abs_softmax import parallel_abs_softmax_1_attn, parallel_abs_softmax_2_attn
from .naive import naive_attn
from .naive_rectified import naive_rectified_attn
from .naive_softpick import naive_softpick_attn
from .naive_relusoftpick import naive_relu_softpick_1_attn, naive_relu_softpick_2_attn
from .naive_abs_softmax import naive_abs_softmax_1_attn, naive_abs_softmax_2_attn
from .naive_gated import naive_gated_attn

__all__ = [
    'parallel_attn',
    'parallel_rectified_attn',
    'parallel_softpick_attn',
    'parallel_relu_softpick_1_attn',
    'parallel_relu_softpick_2_attn',
    'parallel_abs_softmax_1_attn',
    'parallel_abs_softmax_2_attn',
    'naive_attn',
    'naive_rectified_attn',
    'naive_softpick_attn',
    'naive_relu_softpick_1_attn',
    'naive_relu_softpick_2_attn',
    'naive_abs_softmax_1_attn',
    'naive_abs_softmax_2_attn',
    'naive_gated_attn',
]
