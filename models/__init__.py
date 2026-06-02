"""
LP 模型模块
===========

核心模型组件:
- LPModel: 主模型，融合 Libra-MIL 和 PTCMIL 的创新
- CrossAttention: 多头交叉注意力模块
- sinkhorn_ot: Sinkhorn 最优传输算法
- TextEncoder: CONCH 文本编码器
"""

from .lp_model import LPModel
from .cross_attention import CrossAttention
from .optimal_transport import sinkhorn_ot, pairwise_cosine_distance
from .text_encoder import TextEncoder

__all__ = [
    'LPModel',
    'CrossAttention',
    'sinkhorn_ot',
    'pairwise_cosine_distance',
    'TextEncoder',
]
