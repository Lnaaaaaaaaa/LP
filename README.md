# LP: Libra-PTCMIL 融合模型

**LP (Libra-PTCMIL)** 是一个融合了 Libra-MIL 和 PTCMIL 创新点的多实例学习(MIL)模型，用于全切片图像(WSI)分类。

## 核心创新

| 创新点 | 描述 |
|--------|------|
| 非对称 Cross-Attention | 视觉原型为 Query，Patch 为 K/V，复杂度从 O(N²) 降至 O(N·K) |
| 残差动态视觉原型 | 保留初始锚点信息 + 吸收当前 WSI 局部个性 |
| 温度退火软分配 | 训练初期软分配梯度流畅，后期近似硬分配聚类清晰 |
| Patch 级最优传输融合 | 保持 Libra-MIL 的最优传输核心，在 Patch 级别对齐视觉/文本原型 |
| EMA 稳定更新 | 跨 WSI 稳定视觉原型，防止 batch_size=1 导致的剧烈震荡 |

## 项目结构

```
LP/
├── main.py                      # 训练入口
├── models/
│   ├── __init__.py
│   ├── lp_model.py              # LP主模型
│   ├── cross_attention.py       # Cross-Attention模块
│   ├── optimal_transport.py     # Sinkhorn最优传输
│   └── text_encoder.py          # 文本编码器
├── datasets/
│   ├── __init__.py
│   └── dataset.py               # 数据集加载
├── utils/
│   ├── __init__.py
│   ├── general.py               # 通用工具函数
│   └── temperature_scheduler.py # 温度退火调度器
├── text_prompt/                 # 文本提示模板
│   ├── TCGA_RCC_instance_prompt.json
│   ├── TCGA_RCC_bag_prompt.csv
│   ├── TCGA_NSCLC_instance_prompt.json
│   └── TCGA_NSCLC_bag_prompt.csv
├── configs/
│   └── default.yaml             # 默认配置
├── environment.yml              # Conda环境配置
├── .gitignore
└── README.md
```

## 模型架构

```
输入: V_patch (N×D)
        ↓
┌─────────────────────────────────────────────────────────┐
│ 1. 特征投影                                              │
│    V_proj = proj_v(V_patch)                              │
│    P_vis_proj = proj_v(P_vis)  # 正交初始化              │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 动态视觉原型生成 (残差Cross-Attention)                │
│    A_tau = Softmax(Q @ K^T / (tau * √D))    (K×N)       │
│    H_p = P_vis_proj + A_tau @ V_proj        (K×D)        │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 立体最优传输融合                                      │
│    S_v = CosineSim(V_proj, H_p)           (N×K)         │
│    S_t = CosineSim(V_proj, P_text)        (N×K_t)       │
│    T = Sinkhorn(margin_v, margin_t, Cost) (K×K_t)       │
│    S_fused = Σ(S_v @ T) * S_t             (N,)          │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Bag级交叉注意力聚合                                  │
│    V_fused = S_fused * V_proj              (N×D)        │
│    bag_feature = CrossAttn(prompt_bag, V_fused) (D)     │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 5. 分类与损失                                            │
│    logits = MLP(LayerNorm(bag_feature))                 │
│    L = L_cls + α · ||H_p_norm @ H_p_norm^T - I||_F      │
└─────────────────────────────────────────────────────────┘
```

## 环境配置

```bash
conda env create -f environment.yml
conda activate lp
```

## 预训练权重

本项目使用以下预训练权重:
- [CONCH](https://huggingface.co/MahmoodLab/CONCH): 用于文本编码器

下载后放置于 `./conch/pytorch_model.bin`

## 使用方法

### 训练

```bash
python main.py \
    --data_split_json ./data_4shot/data_split.json \
    --data_csv ./data_4shot/labels.csv \
    --h5_file_dir /mnt/sda2/WSI/muti-modal/TCGA-RCC-fea/features \
    --instance_prompt ./text_prompt/TCGA_RCC_instance_prompt.json \
    --bag_prompt ./text_prompt/TCGA_RCC_bag_prompt.csv \
    --text_model_weights_path  /mnt/sda1/ln_workspace/CONCH/checkpoints/pytorch_model.bin \
    --save_dir  ./results4/TCGA_RCC_k=10 \
    --K 10 \
    --num_classes 3 \
    --epochs 20
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--K` | 视觉原型数量 | 4 |
| `--K_t` | 文本原型数量 | 4 |
| `--tau_init` | 初始温度 | 1.0 |
| `--tau_min` | 最小温度 | 0.05 |
| `--alpha_ptc` | PTC损失权重 | 0.1 |
| `--ema_momentum` | EMA动量 | 0.9 |
| `--lr` | 学习率 | 1e-4 |

## 与原始方法的对比

| 特性 | Libra-MIL | PTCMIL | LP |
|------|-----------|--------|-----|
| 视觉原型 | 随机初始化可学习 | 正交初始化 + PTC损失 | 正交初始化 + 残差更新 + PTC损失 |
| 聚类机制 | 无 | Prompt Token聚类 | Cross-Attention聚类 |
| 全局交互 | 无 | O(N²) Self-Attention | O(N·K) Cross-Attention |
| 文本引导 | 有 (CONCH) | 无 | 有 (CONCH) |
| 最优传输 | Sinkhorn | 无 | Sinkhorn (Patch级) |
| 动态更新 | 无 | Momentum更新 | EMA更新 |

## 消融实验建议

| 实验 | 对照组 | 实验组 | 验证目标 |
|------|--------|--------|----------|
| 1 | 无温度退火 (τ=1.0) | 温度退火 | 软→硬分配的效果 |
| 2 | 无PTC损失 (α=0) | PTC损失 (α=0.1) | 正交性约束的必要性 |
| 3 | 无残差连接 | 残差连接 | 初始锚点保留的作用 |
| 4 | 无EMA | EMA | 跨WSI稳定性的效果 |
| 5 | K=4 | K=6/8 | 聚类数量对性能的影响 |

## 参考文献

1. Libra-MIL: Multimodal Prototypes Stereoscopic Infused with Task-specific Language Priors for Few-shot WSI Classification
2. PTCMIL: Multiple Instance Learning via Prompt Token Clustering for Whole Slide Image Classification
3. CONCH: Contrastive Learning for Clinical Histopathology (Lu et al. 2024, Nature Medicine)

## 许可证

MIT License
