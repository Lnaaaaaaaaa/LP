"""
最优传输模块
============

实现 Sinkhorn-Knopp 算法求解最优传输问题

核心概念:
    - 最优传输: 找到将一个分布转换为另一个分布的最优方案
    - Sinkhorn 算法: 通过熵正则化高效求解最优传输

参考:
    - Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport
"""

import torch
import torch.nn.functional as F


def pairwise_cosine_distance(x, y):
    """
    计算两组向量之间的成对余弦距离

    参数:
        x: (N, D) 第一组向量
        y: (M, D) 第二组向量

    返回:
        distance: (N, M) 余弦距离矩阵
                  distance[i,j] = 1 - cos(x[i], y[j])

    数学公式:
        cos_sim = (x · y) / (||x|| * ||y||)
        cos_dist = 1 - cos_sim

    说明:
        - 余弦距离范围: [0, 2]
        - 当两向量方向相同时，距离=0
        - 当两向量方向相反时，距离=2
        - 当两向量正交时，距离=1
    """
    # L2归一化
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # 计算余弦相似度矩阵，然后取反
    return 1.0 - torch.matmul(x, y.t())


def sinkhorn_ot(mu, nu, cost, epsilon=0.05, n_iters=20):
    """
    批量版本的 Sinkhorn 最优传输

    参数:
        mu: (B, N, K1) 源分布
        nu: (B, N, K2) 目标分布
        cost: (K1, K2) 代价矩阵
        epsilon: 熵正则化系数
        n_iters: 迭代次数

    返回:
        T: (B, N, K1, K2) 最优传输矩阵

    说明:
        批量版本用于同时对多个样本求解最优传输
        主要用于继承 Libra-MIL 的原始实现风格
    """
    B, N, K1 = mu.shape
    _, _, K2 = nu.shape

    # 扩展代价矩阵到batch维度: (K1, K2) -> (B, N, K1, K2)
    cost = cost.unsqueeze(0).unsqueeze(0).expand(B, N, K1, K2)

    # 计算核矩阵 K = exp(-C/ε)
    K_mat = torch.exp(-cost / epsilon)

    # 初始化对偶变量 u, v
    u = torch.ones_like(mu) / K1
    v = torch.ones_like(nu) / K2

    # Sinkhorn迭代
    for _ in range(n_iters):
        # 行归一化
        u = mu / (torch.einsum("bnij,bnj->bni", K_mat, v) + 1e-8)
        # 列归一化
        v = nu / (torch.einsum("bnij,bni->bnj", K_mat, u) + 1e-8)

    # 计算最终的传输矩阵
    T = K_mat * u.unsqueeze(-1) * v.unsqueeze(-2)

    return T
