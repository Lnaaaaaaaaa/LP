"""
LP 主模型模块
=============

融合 Libra-MIL 和 PTCMIL 的多实例学习模型

核心创新点:
    1. 非对称 Cross-Attention
       - 视觉原型为 Query，Patch 为 K/V
       - 复杂度从 O(N²) 降至 O(N·K)

    2. 残差动态视觉原型
       - 保留初始锚点信息 + 吸收当前 WSI 局部个性
       - H_p = P_vis_proj + A_tau @ V_proj

    3. 温度退火软分配
       - 训练初期软分配梯度流畅
       - 后期近似硬分配聚类清晰

    4. 实例级最优传输融合 (关键改进)
       - 对每个Patch独立进行最优传输对齐
       - 返回 (N, K) 的注意力矩阵，保留原型维度信息
       - 避免信息瓶颈，与Libra-MIL对齐

    5. EMA 稳定更新
       - 跨 WSI 稳定视觉原型
       - 防止 batch_size=1 导致的剧烈震荡

参考:
    - Libra-MIL: Multimodal Prototypes Stereoscopic Infused with Task-specific Language Priors
    - PTCMIL: Multiple Instance Learning via Prompt Token Clustering
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import CrossAttention, TemperatureScaledCrossAttention
from .optimal_transport import sinkhorn_ot, sinkhorn_ot_batch, pairwise_cosine_distance


def init_visual_prototypes(K, dim):
    """
    初始化正交视觉原型

    使用 PyTorch 原生正交初始化（基于 QR 分解）

    参数:
        K: 视觉原型数量
        dim: 特征维度

    返回:
        P_vis: (K, dim) 正交初始化的视觉原型

    说明:
        正交初始化确保不同聚类的中心相互独立
        比手写 Gram-Schmidt 更稳定高效
    """
    P_vis = torch.empty(K, dim)
    nn.init.orthogonal_(P_vis)
    return P_vis


class LPModel(nn.Module):
    """
    LP 主模型类

    模型架构:
        输入: V_patch (N×D)
            ↓
        1. 特征投影
           V_proj = proj_v(V_patch)
           P_vis_proj = proj_v(P_vis)
            ↓
        2. 动态视觉原型生成 (残差Cross-Attention)
           A_tau = Softmax(Q @ K^T / (tau * √D))    (K×N)
           H_p = P_vis_proj + A_tau @ V_proj        (K×D)
            ↓
        3. 实例级最优传输融合
           S_v = CosineSim(V_proj, H_p) / tau      (N×K)
           S_t = CosineSim(V_proj, P_text) / tau   (N×K_t)
           attn_v = Softmax(S_v, dim=-1)           (N×K)
           attn_t = Softmax(S_t, dim=-1)           (N×K_t)
           T = Sinkhorn(attn_v, attn_t, Cost)      (N×K×K_t)
           attn_fused = Σ_j T[:,:,j]               (N×K)
            ↓
        4. 特征加权（保留原型维度）
           V_fused = Σ_k attn_fused[:,:,k] * H_p[k]  (N×D)
            ↓
        5. Bag级交叉注意力聚合
           bag_feature = CrossAttn(prompt_bag, V_fused) (D)
            ↓
        6. 分类与损失
           logits = MLP(LayerNorm(bag_feature))
           L = L_cls + α · ||H_p_norm @ H_p_norm^T - I||_F

    参数:
        dim: 特征维度，默认512
        K: 视觉原型数量，默认4
        K_t: 文本原型数量，默认4
        num_classes: 类别数，默认2
        P_text: (K_t, dim) 预计算的文本原型特征
        prompt_bag: (C, dim) 预计算的Bag级文本原型特征
        tau_init: 初始温度，默认1.0
        tau_min: 最小温度，默认0.05
        ema_momentum: EMA动量，默认0.9
        alpha_ptc: PTC损失权重，默认0.1
        ot_epsilon: 最优传输熵正则化系数，默认0.05
        ot_iters: Sinkhorn迭代次数，默认20
        num_heads: 注意力头数，默认8
        dropout: Dropout概率，默认0.1
    """

    def __init__(
        self,
        dim=512,
        K=4,
        K_t=4,
        num_classes=2,
        P_text=None,
        prompt_bag=None,
        tau_init=1.0,
        tau_min=0.05,
        ema_momentum=0.9,
        alpha_ptc=0.1,
        ot_epsilon=0.05,
        ot_iters=20,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        # ========== 保存超参数 ==========
        self.dim = dim
        self.K = K                    # 视觉原型数量
        self.K_t = K_t                # 文本原型数量
        self.num_classes = num_classes
        self.ema_momentum = ema_momentum
        self.alpha_ptc = alpha_ptc
        self.ot_epsilon = ot_epsilon
        self.ot_iters = ot_iters
        self.tau_min = tau_min

        # ========== 文本原型 (buffer，不参与梯度计算) ==========
        if P_text is not None:
            self.register_buffer("P_text", P_text)
        else:
            # 如果没有提供，使用随机初始化（实际使用时应该从CONCH加载）
            self.register_buffer("P_text", torch.randn(K_t, dim))

        if prompt_bag is not None:
            self.register_buffer("prompt_bag", prompt_bag)
        else:
            self.register_buffer("prompt_bag", torch.randn(num_classes, dim))

        # ========== 视觉原型（可训练参数）==========
        # 使用正交初始化
        self.P_vis = nn.Parameter(init_visual_prototypes(K, dim))

        # EMA 状态（buffer，不参与梯度计算）
        self.register_buffer('P_vis_ema', self.P_vis.data.clone())

        # ========== 投影层 ==========
        # 将特征映射到统一的原型空间
        self.proj_v = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        

        # ========== 动态视觉原型生成的 Cross-Attention ==========
        # 使用温度缩放版本
        self.prototype_cross_attn = TemperatureScaledCrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # ========== Bag级交叉注意力聚合 ==========
        self.bag_cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # ========== 分类头 ==========
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * num_classes),
            nn.Linear(dim * num_classes, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

        # ========== 用于PTC损失的单位矩阵 ==========
        self.register_buffer('identity_matrix', torch.eye(K))

    def compute_dynamic_prototypes(self, V_proj, tau):
        """
        计算动态视觉原型

        参数:
            V_proj: (N, D) 投影后的Patch特征
            tau: 温度参数

        返回:
            H_p: (K, D) 动态视觉原型
            attn_weights: (K, N) 注意力权重矩阵

        数学公式:
            A_tau = Softmax(P_vis_proj @ V_proj^T / (tau * √D))
            H_p = P_vis_proj + A_tau @ V_proj

        设计要点:
            - 残差连接: 保留初始锚点信息
            - 温度缩放: 控制聚类锐度
        """
        # 选择当前使用的视觉原型
        # 训练时使用带梯度的参数，推理时使用EMA平滑版本
        if self.training:
            P_vis_current = self.P_vis
        else:
            P_vis_current = self.P_vis_ema

        # 投影视觉原型
        P_vis_proj = self.proj_v(P_vis_current)  # (K, D)

        # 使用 Cross-Attention 计算动态原型
        # Query: 视觉原型, Key/Value: Patch特征
        H_p, attn_weights = self.prototype_cross_attn(
            query=P_vis_proj,    # (K, D)
            key=V_proj,          # (N, D)
            value=V_proj,        # (N, D)
            tau=tau
        )

        # 残差连接
        H_p = P_vis_proj + H_p

        return H_p, attn_weights

    def compute_fused_weights(self, V_proj, H_p, tau):
        """
        计算融合权重（实例级最优传输版本）

        参数:
            V_proj: (N, D) 投影后的Patch特征
            H_p: (K, D) 动态视觉原型
            tau: 温度参数

        返回:
            attn_fused: (N, K) 每个Patch对每个视觉原型的融合权重

        数学公式:
            S_v = CosineSim(V_proj, H_p) / tau     # (N, K)
            S_t = CosineSim(V_proj, P_text) / tau  # (N, K_t)
            attn_v = Softmax(S_v, dim=-1)          # (N, K)
            attn_t = Softmax(S_t, dim=-1)          # (N, K_t)
            Cost = CosineDist(H_p, P_text)         # (K, K_t)
            T = Sinkhorn(attn_v, attn_t, Cost)     # (N, K, K_t) 实例级OT
            attn_fused = Σ_j T[:,:,j]              # (N, K)

        关键改进:
            - 返回 (N, K) 而不是 (N,)，保留原型维度信息
            - 使用实例级最优传输，每个Patch独立对齐
        """
        N = V_proj.size(0)

        # 投影文本原型
       #P_text_proj = self.proj_text(self.P_text)  # (K_t, D)

        # 计算相似度矩阵（带温度缩放）
        # S_v[i,k] = 第i个Patch对第k个视觉原型的相似度
        S_v = (1.0 - pairwise_cosine_distance(V_proj, H_p)) / tau  # (N, K)

        # S_t[i,j] = 第i个Patch对第j个文本原型的相似度
        S_t = (1.0 - pairwise_cosine_distance(V_proj, self.P_text)) / tau  # (N, K_t)

        # 转换为注意力分布
        # attn_v = F.softmax(S_v, dim=-1)  # (N, K)
        # attn_t = F.softmax(S_t, dim=-1)  # (N, K_t)
        
        # 【修复2】：计算全局边缘分布 (Marginals)
        margin_v = F.softmax(S_v.mean(dim=0), dim=-1)  # (K,)
        margin_t = F.softmax(S_t.mean(dim=0), dim=-1)  # (K_t,)

        # 计算视觉原型与文本原型之间的代价矩阵
        Cost = pairwise_cosine_distance(H_p, self.P_text)  # (K, K_t)

        # 实例级最优传输
        # 扩展到batch维度: (N, K) -> (1, N, K), (N, K_t) -> (1, N, K_t)
        # attn_v_batch = attn_v.unsqueeze(0)  # (1, N, K)
        # attn_t_batch = attn_t.unsqueeze(0)  # (1, N, K_t)

        # 使用批量Sinkhorn算法
        T = sinkhorn_ot_batch(margin_v, margin_t, Cost,
                              self.ot_epsilon, self.ot_iters)  # (1, N, K, K_t)
        
        S_fused = torch.sum((S_v @ T) * S_t, dim=-1)  # (N,)
    
        return S_fused
       

    def forward(self, V_patch, labels=None, tau=None):
        """
        前向传播

        参数:
            V_patch: (B, N, D) 或 (N, D) 输入特征
                     B: batch size (通常为1)
                     N: patch数量
                     D: 特征维度 (512)
            labels: (B,) 标签 (可选，训练时提供)
            tau: 温度参数 (可选，默认使用tau_min)

        返回:
            dict: {
                'logits': (B, num_classes) 分类logits,
                'loss': 损失值 (如果提供labels),
                'H_p': 动态视觉原型,
                'S_fused': 融合权重,
            }

        注意:
            EMA 更新在训练循环的 optimizer.step() 之后进行
            不在 forward 内部更新，避免切断梯度图
        """
        # 处理输入维度
        squeeze_output = False
        if V_patch.dim() == 2:
            V_patch = V_patch.unsqueeze(0)
            squeeze_output = True

        B, N, D = V_patch.shape

        # 默认使用最小温度（推理时）或由外部传入（训练时）
        if tau is None:
            tau = self.tau_min

        # ========== 1. 特征投影 ==========
        V_proj = self.proj_v(V_patch)  # (B, N, D)

        # ========== 2. 动态视觉原型生成 ==========
        H_p_list = []
        attn_fused_list = []

        for b in range(B):
            H_p, attn_weights = self.compute_dynamic_prototypes(V_proj[b], tau)
            # S_fused 是 (N,) 的标量权重
            S_fused = self.compute_fused_weights(V_proj[b], H_p, tau)
            H_p_list.append(H_p)
            attn_fused_list.append(S_fused)

        H_p = torch.stack(H_p_list, dim=0)  # (B, K, D)
        attn_fused = torch.stack(attn_fused_list, dim=0)  # (B, N, K)

        # ========== 3. 特征加权（保留原型维度）==========
        # # 方法1：使用注意力加权聚合原型信息
        # # V_fused[b,n,:] = Σ_k attn_fused[b,n,k] * H_p[b,k,:]
        # V_fused = torch.einsum('bnk,bkd->bnd', attn_fused, H_p)  # (B, N, D)
        
        # 【修复3】：使用标量权重对 Patch 特征进行加权，代替原来的张量 einsum
        # 这极大降低了特征学习的难度
        V_fused = S_fused_batch.unsqueeze(-1) * V_proj  # (B, N, D)

        # 方法2（可选）：残差连接原始特征
        # V_fused = V_fused + V_proj  # 如果效果不好可以尝试加上

        # ========== 4. Bag级交叉注意力聚合 ==========
        # # prompt_bag: (C, D) -> (B, C, D)
        # prompt_bag = self.proj_text(self.prompt_bag).unsqueeze(0).expand(B, -1, -1)
        
        # Bag 级聚合
        prompt_bag = self.prompt_bag.unsqueeze(0).expand(B, -1, -1) # 直接使用，不投影

        bag_feature = self.bag_cross_attn(
            query=prompt_bag,   # (B, C, D)
            key=V_fused,        # (B, N, D)
            value=V_fused       # (B, N, D)
        )  # -> (B, C, D)

        # # Bag特征聚合（平均池化）
        # bag_feature_pooled = bag_feature.mean(dim=1)  # (B, D)
        
        # 【修复4】：展平代替平均池化，保留类别专属语义
        bag_feature_flat = bag_feature.view(B, -1)  # (B, C * D)

        # ========== 5. 分类 ==========
        logits = self.classifier(bag_feature_flat)  # (B, num_classes)

        # 构建输出字典
        output = {
            'logits': logits,
            'H_p': H_p,
            'attn_fused': attn_fused,
        }

        # ========== 6. 计算损失 ==========
        if labels is not None:
            # 分类损失
            loss_cls = F.cross_entropy(logits, labels)

            # PTC损失：约束动态原型正交
            H_p_norm = F.normalize(H_p, p=2, dim=-1)
            # 对每个batch计算PTC损失
            ptc_losses = []
            for b in range(B):
                ptc_loss = torch.norm(
                    H_p_norm[b] @ H_p_norm[b].T - self.identity_matrix,
                    p='fro'
                )
                ptc_losses.append(ptc_loss)
            loss_ptc = torch.stack(ptc_losses).mean()

            # 总损失
            loss = loss_cls + self.alpha_ptc * loss_ptc

            output['loss'] = loss
            output['loss_cls'] = loss_cls
            output['loss_ptc'] = loss_ptc

        # 恢复输出维度
        if squeeze_output:
            output['logits'] = output['logits'].squeeze(0)
            output['H_p'] = output['H_p'].squeeze(0)
            output['attn_fused'] = output['attn_fused'].squeeze(0)
            
            
        

        return output

    def update_ema(self):
        """
        更新 EMA 视觉原型

        ⚠️ 关键：此方法必须在 optimizer.step() 之后调用
        在训练循环中：
            optimizer.step()
            model.update_ema()  # 更新EMA

        数学公式:
            P_vis_ema = momentum * P_vis_ema + (1 - momentum) * P_vis
        """
        with torch.no_grad():
            self.P_vis_ema.copy_(
                self.ema_momentum * self.P_vis_ema +
                (1 - self.ema_momentum) * self.P_vis.data
            )

    def get_tau_for_epoch(self, epoch, total_epochs, tau_init=1.0, decay_rate=0.95):
        """
        根据训练轮次计算温度参数

        参数:
            epoch: 当前轮次
            total_epochs: 总轮次
            tau_init: 初始温度
            decay_rate: 衰减率

        返回:
            tau: 当前温度

        退火曲线:
            Epoch 0: τ = 1.00 (完全软分配)
            Epoch 30: τ ≈ 0.21 (聚类明确)
            Epoch 60+: τ = 0.05 (最小值锁定)
        """
        tau = tau_init * (decay_rate ** epoch)
        tau = max(tau, self.tau_min)
        return tau
