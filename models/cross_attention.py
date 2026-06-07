"""
交叉注意力模块
==============

实现多头交叉注意力机制，用于：
1. 动态视觉原型生成（模块2）
2. Bag级特征聚合（模块4）

核心概念:
    - Cross-Attention: Query 来自一个模态，Key/Value 来自另一个模态
    - Multi-Head: 将特征分成多个头并行处理，捕捉不同的关联模式

复杂度优势:
    - 标准 Self-Attention: O(N² · D)
    - Cross-Attention (Q=K, V=N): O(N · K · D)
    - 当 K << N 时，计算效率显著提升

参考:
    - Vaswani et al. (2017). Attention Is All You Need
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    多头交叉注意力模块

    用于两种场景:
        1. 动态视觉原型生成:
           - Query: 视觉原型 (K, D)
           - Key/Value: Patch特征 (N, D)
           - 复杂度: O(N · K · D)

        2. Bag级特征聚合:
           - Query: Bag级文本原型 (C, D)
           - Key/Value: 融合后的Patch特征 (N, D)
           - 复杂度: O(N · C · D)

    参数:
        dim: 特征维度
        num_heads: 注意力头数，默认8
        dropout: Dropout概率，默认0.1
        qkv_bias: 是否使用偏置，默认True
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子 1/√d_k

        # 线性投影层
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.o_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask=None):
        """
        前向传播

        参数:
            query: (B, Q, D) 查询向量
            key: (B, K, D) 键向量
            value: (B, K, D) 值向量
            attention_mask: 可选的注意力掩码

        返回:
            output: (B, Q, D) 注意力输出

        数学公式:
            Attention(Q, K, V) = softmax(QK^T / √d_k) * V

        处理流程:
            1. 线性投影 Q, K, V
            2. 重塑为多头格式
            3. 计算缩放点积注意力
            4. 加权求和
            5. 输出投影
        """
        B, Q, D = query.shape
        _, K, _ = key.shape

        # 线性投影
        Q_proj = self.q_proj(query)
        K_proj = self.k_proj(key)
        V_proj = self.v_proj(value)

        # 重塑为多头格式: (B, num_heads, seq_len, head_dim)
        Q_proj = Q_proj.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = K_proj.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        V_proj = V_proj.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算缩放点积注意力
        # 使用 PyTorch 内置的优化实现，支持 flash attention 加速
        attn_output = F.scaled_dot_product_attention(
            Q_proj, K_proj, V_proj,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        # 重塑回原始格式: (B, Q, D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, Q, D)

        # 输出投影
        output = self.o_proj(attn_output)

        return output


class TemperatureScaledCrossAttention(nn.Module):
    """
    温度缩放的交叉注意力模块

    在标准 CrossAttention 基础上添加可调节的温度参数，
    用于控制注意力分布的锐度。

    温度参数的作用:
        - 低温 (τ小): 注意力分布更尖锐，更集中于最相关的token
        - 高温 (τ大): 注意力分布更平滑，更均匀地关注所有token

    应用场景:
        动态视觉原型生成时，使用温度退火策略：
        - 训练初期: 高温，软分配，梯度流畅
        - 训练后期: 低温，近似硬分配，聚类清晰
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, tau=1.0):
        """
        前向传播

        参数:
            query: (B, Q, D) 或 (Q, D) 查询向量
            key: (B, K, D) 或 (K, D) 键向量
            value: (B, K, D) 或 (K, D) 值向量
            tau: 温度参数，默认1.0

        返回:
            output: 注意力输出
            attn_weights: 注意力权重矩阵 (用于可视化)
        """
        # 处理无batch维度的情况
        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_output = True

        B, Q, D = query.shape
        _, K, _ = key.shape

        # 线性投影
        Q_proj = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = self.k_proj(key).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        V_proj = self.v_proj(value).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = (Q_proj @ K_proj.transpose(-2, -1)) * self.scale / tau

        # Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = attn_weights @ V_proj
        output = output.transpose(1, 2).reshape(B, Q, D)
        output = self.o_proj(output)

        if squeeze_output:
            output = output.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return output, attn_weights  
    #  output = 所有 patch 特征的加权平均，权重高的贡献大，但不是只取那一个
