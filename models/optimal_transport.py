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


def sinkhorn_ot(a, b, C, epsilon=0.05, n_iters=20):
    """
    Sinkhorn-Knopp 算法求解最优传输

    功能:
        在两个分布之间找到最优传输方案（耦合矩阵）

    参数:
        a: (K,) 源边缘分布（视觉原型边缘）
        b: (K_t,) 目标边缘分布（文本原型边缘）
        C: (K, K_t) 代价矩阵，通常是原型间的余弦距离
        epsilon: 熵正则化系数，控制传输方案的平滑程度
                 - 较小的 epsilon → 更锐利的传输
                 - 较大的 epsilon → 更平滑的传输
        n_iters: Sinkhorn 迭代次数

    返回:
        T: (K, K_t) 最优传输矩阵
           T[i,j] 表示从源分布 i 传输到目标分布 j 的量

    算法原理:
        最优传输问题:
            min_T <T, C> - ε*H(T)
            s.t. T·1 = a, T^T·1 = b

        其中:
        - <T, C> 是传输代价（Frobenius内积）
        - H(T) 是熵正则项，使传输方案更平滑
        - a, b 是边缘分布约束

        Sinkhorn算法通过交替归一化求解:
        1. u = a / (K_kernel @ v)
        2. v = b / (K_kernel^T @ u)
        其中 K_kernel = exp(-C/ε) 是核矩阵

    直观理解:
        想象有一批货物(a)要运送到目的地(b)
        代价矩阵C决定运输成本
        Sinkhorn算法找到成本最低的运输方案T

    使用示例:
        >>> a = torch.tensor([0.5, 0.5])  # 源分布
        >>> b = torch.tensor([0.3, 0.7])  # 目标分布
        >>> C = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # 代价矩阵
        >>> T = sinkhorn_ot(a, b, C)
        >>> # T[0,0] 表示从源0运到目标0的量
    """
    # 确保输入是一维的
    if a.dim() > 1:
        a = a.squeeze()
    if b.dim() > 1:
        b = b.squeeze()

    # 计算核矩阵 K = exp(-C/ε)
    # 代价越小，核值越大，传输概率越高
    # 使用 K_kernel 避免与视觉原型数量 K 混淆
    K_kernel = torch.exp(-C / epsilon)

    # 初始化对偶变量 u, v
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # Sinkhorn迭代
    for _ in range(n_iters):
        # 列归一化：使 T.sum(dim=0) = b
        v = b / (K_kernel.T @ u + 1e-8)
        # 行归一化：使 T.sum(dim=1) = a
        u = a / (K_kernel @ v + 1e-8)

    # 计算最终的传输矩阵
    # T = diag(u) @ K_kernel @ diag(v)
    T = u.unsqueeze(-1) * K_kernel * v.unsqueeze(0)

    return T


def sinkhorn_ot_batch(mu, nu, cost, epsilon=0.05, n_iters=20):
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
