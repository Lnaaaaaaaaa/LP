"""
LP 工具函数模块
===============

包含模型训练和评估所需的通用工具函数：
- compute_metrics: 计算分类指标（准确率、AUC、F1分数）
- CSVWriter: CSV日志写入器
- write_summary_log: 汇总多折交叉验证结果
- set_seed: 设置随机种子
- EarlyStopping: 早停机制
"""

import os
import csv
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
from typing import List, Tuple, Union


def set_seed(seed=42):
    """
    设置随机种子，确保实验可复现

    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: Union[List[float], List[List[float]], np.ndarray],
    num_classes: int = 2
) -> Tuple[float, float, float]:
    """
    计算分类任务的评估指标

    参数:
        y_true: 真实标签列表，shape: (N,)
        y_pred: 预测标签列表，shape: (N,)
        y_score: 预测概率/分数，shape: (N,) 或 (N, num_classes)
        num_classes: 类别数量，默认为2（二分类）

    返回:
        acc: 准确率 (Accuracy)
        auc: ROC曲线下面积 (Area Under Curve)
        f1: F1分数 (精确率和召回率的调和平均)
    """
    # 计算准确率
    acc = accuracy_score(y_true, y_pred)

    y_score = np.array(y_score)

    if num_classes == 2:
        # 二分类
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            pos_probs = y_score[:, 1]
        else:
            pos_probs = y_score
        auc = roc_auc_score(y_true, pos_probs)
    else:
        # 多分类
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        auc = roc_auc_score(y_true_bin, y_score, multi_class='ovr')

    # 计算F1分数
    average_mode = 'binary' if num_classes == 2 else 'macro'
    f1 = f1_score(y_true, y_pred, average=average_mode)

    return acc, auc, f1


class CSVWriter:
    """
    CSV日志写入器

    用于记录训练过程中的指标和结果

    使用示例:
        writer = CSVWriter('log.csv', header=['epoch', 'loss', 'acc'])
        writer.write_row([1, 0.5, 0.8])
        writer.write_rows([[2, 0.4, 0.85], [3, 0.3, 0.9]])
    """

    def __init__(self, filename, header=None, sep=',', append=False):
        """
        初始化CSV写入器

        参数:
            filename: CSV文件路径
            header: 表头列表，如 ['epoch', 'loss', 'acc']
            sep: 分隔符，默认为逗号
            append: 是否追加模式
        """
        self.filename = filename
        self.sep = sep

        if Path(self.filename).exists() and not append:
            os.remove(self.filename)

        if header is not None:
            self.write_row(header)

    def write_row(self, row):
        """写入单行数据"""
        with open(self.filename, 'a+', newline='') as fp:
            csv_writer = csv.writer(fp, delimiter=self.sep)
            csv_writer.writerow(row)
            fp.flush()

    def write_rows(self, rows):
        """写入多行数据"""
        with open(self.filename, 'a+', newline='') as fp:
            csv_writer = csv.writer(fp, delimiter=self.sep)
            csv_writer.writerows(rows)
            fp.flush()


def write_summary_log(final_log_path, summary_log_path=None):
    """
    汇总多折交叉验证的结果

    参数:
        final_log_path: 各折结果的CSV文件路径
        summary_log_path: 汇总结果保存路径
    """
    if summary_log_path is None:
        summary_log_path = os.path.join(os.path.dirname(final_log_path), 'summary_log.csv')

    df = pd.read_csv(final_log_path)
    df = df[pd.to_numeric(df.get('fold', df.get('flod', df.iloc[:, 0])), errors='coerce').notnull()]

    metric_cols = [col for col in df.columns if 'acc' in col.lower() or 'auc' in col.lower() or 'f1' in col.lower()]

    if metric_cols:
        mean_vals = df[metric_cols].mean()
        std_vals = df[metric_cols].std()

        summary_df = pd.DataFrame([
            ['mean'] + mean_vals.round(4).tolist(),
            ['std'] + std_vals.round(4).tolist()
        ], columns=['metric'] + metric_cols)

        summary_df.to_csv(summary_log_path, index=False)


class EarlyStopping:
    """
    早停机制

    当验证指标在指定轮次内没有改善时，停止训练

    参数:
        patience: 容忍轮次
        delta: 最小改善阈值
        mode: 'max' 或 'min'，指示指标是越大越好还是越小越好
    """

    def __init__(self, patience=15, delta=0.0, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        检查是否应该早停

        参数:
            score: 当前验证指标

        返回:
            bool: 是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
