"""
文本编码器模块
==============

使用 CONCH 模型将病理学文本描述编码为特征向量

CONCH 简介:
    - CONCH (CONtrastive learning for Clinical Histopathology)
    - 微软开发的病理学专用视觉-语言基础模型
    - 论文: Lu et al. 2024, Nature Medicine
    - 预训练于大规模病理图像-文本对

为什么使用 CONCH:
    1. 针对病理学领域优化，理解医学术语
    2. 视觉和文本特征在同一语义空间对齐
    3. 支持零样本分类和少样本学习

使用示例:
    encoder = TextEncoder(output_dim=512)
    texts = ["clear cell carcinoma", "papillary renal cell carcinoma"]
    features = encoder(texts)  # shape: (2, 512)
"""

import torch
import torch.nn as nn
import json
import pandas as pd


class TextEncoder(nn.Module):
    """
    文本编码器类

    功能:
        将病理学文本描述编码为512维特征向量

    模型架构:
        - 基于ViT-B-16 (Vision Transformer - Base, patch size 16)
        - 输出维度: 512 (与视觉特征维度对齐)

    参数:
        output_dim: 输出特征维度，默认512
        weights_path: CONCH模型权重路径，默认 "conch/pytorch_model.bin"
    """

    def __init__(self, output_dim=512, weights_path="conch/pytorch_model.bin"):
        super().__init__()
        self.output_dim = output_dim
        self.weights_path = weights_path

        # 延迟加载模型，避免在没有CONCH时出错
        self._model_loaded = False
        self.base_model = None
        self.tokenizer = None
        self.tokenize_func = None

    def _load_model(self):
        """延迟加载CONCH模型"""
        if self._model_loaded:
            return

        try:
            from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
            self.tokenizer = get_tokenizer()
            self.base_model, _ = create_model_from_pretrained(
                'conch_ViT-B-16',
                self.weights_path
            )
            self.tokenize_func = tokenize
            self._model_loaded = True
        except ImportError as e:
            raise ImportError(
                "CONCH 模块未找到。请确保 CONCH 已正确安装。\n"
                "安装方法: pip install git+https://github.com/mahmoodlab/CONCH.git\n"
                f"错误信息: {e}"
            )

    def forward(self, texts):
        """
        编码文本

        参数:
            texts: 文本列表，例如 ["clear cell carcinoma", "papillary carcinoma"]

        返回:
            text_feats: 文本特征张量，shape [N, output_dim]
                        N是文本数量，output_dim是特征维度(512)

        处理流程:
            1. tokenize: 将文本转为token ID序列
            2. encode_text: 通过CONCH的文本编码器提取特征
            3. 返回特征向量（已归一化到单位球面）

        注意:
            - 使用 torch.no_grad() 禁用梯度计算
            - 因为CONCH是冻结的，不参与训练
        """
        self._load_model()

        # 文本分词：将字符串转换为模型可处理的token序列
        tokenized = self.tokenize_func(texts=texts, tokenizer=self.tokenizer)

        # 使用CONCH编码文本（不计算梯度，模型冻结）
        with torch.no_grad():
            text_feats = self.base_model.encode_text(tokenized)

        return text_feats


def load_text_prompts(instance_prompt_path, bag_prompt_path, text_encoder, device='cuda'):
    """
    加载并编码文本原型

    参数:
        instance_prompt_path: 实例级文本提示 JSON 文件路径
        bag_prompt_path: Bag级文本提示 CSV 文件路径
        text_encoder: 文本编码器实例
        device: 计算设备

    返回:
        P_text: (K_t, D) 实例级文本原型特征
        prompt_bag: (C, D) Bag级文本原型特征

    说明:
        - 实例级文本原型用于最优传输
        - Bag级文本原型用于最终聚合
    """
    # 加载实例级文本提示
    with open(instance_prompt_path, 'r', encoding='utf-8') as f:
        instance_prompts = json.load(f)

    # 加载Bag级文本提示
    bag_prompts_df = pd.read_csv(bag_prompt_path)

    # 编码文本原型
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    with torch.no_grad():
        # 实例级文本原型
        if isinstance(instance_prompts, dict):
            instance_texts = list(instance_prompts.values())
        else:
            instance_texts = instance_prompts
        P_text = text_encoder(instance_texts)

        # Bag级文本原型
        if 'prompt' in bag_prompts_df.columns:
            bag_texts = bag_prompts_df['prompt'].tolist()
        else:
            bag_texts = bag_prompts_df.iloc[:, 0].tolist()
        prompt_bag = text_encoder(bag_texts)

    return P_text, prompt_bag
