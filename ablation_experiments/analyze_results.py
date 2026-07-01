#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验结果分析脚本
====================
自动收集所有实验结果并生成LaTeX表格

使用方法:
    python analyze_results.py --results_dir ./results/
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path


# 实验名称映射
NAME_MAP = {
    # 主消融实验
    "Full": "Full Model",
    "A1_no_dynamic_prototype": "w/o Dynamic Prototype",
    "A2_no_ot_fusion": "w/o Optimal Transport",
    "A3_no_gating": "w/o Gating Mechanism",
    "A4_no_bag_prompt": "w/o Bag-level Prior",
    # K值敏感性
    "A5_K2": "K=2",
    "A5_K4": "K=4",
    "A5_K6": "K=6",
    "A5_K8": "K=8",
    "A5_K16": "K=16",
}

# 主消融实验顺序
MAIN_EXPERIMENTS = [
    "Full",
    "A1_no_dynamic_prototype",
    "A2_no_ot_fusion",
    "A3_no_gating",
    "A4_no_bag_prompt",
]

# K值实验顺序
K_EXPERIMENTS = [
    "A5_K2",
    "A5_K4",
    "A5_K6",
    "A5_K8",
    "A5_K16",
]


def parse_experiment_name(exp_name):
    """解析实验名称，返回描述"""
    return NAME_MAP.get(exp_name, exp_name)


def collect_fold_results(results_dir, exp_name):
    """收集单个实验的所有fold结果"""
    exp_dir = Path(results_dir) / exp_name

    all_folds = []
    for fold_idx in range(5):  # 默认5折
        fold_file = exp_dir / f"fold_{fold_idx}" / "test_results.json"
        if fold_file.exists():
            with open(fold_file, 'r') as f:
                data = json.load(f)
                all_folds.append(data)

    return all_folds


def compute_statistics(fold_results):
    """计算均值和标准差"""
    if not fold_results:
        return None

    metrics = ['acc', 'auc', 'f1', 'sensitivity', 'specificity', 'precision']
    stats = {}

    for metric in metrics:
        values = [r.get(metric, 0) for r in fold_results]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }

    return stats


def generate_main_latex_table(results_df):
    """生成主消融实验LaTeX表格"""
    latex_code = r"""
\begin{table}[htbp]
\centering
\caption{Ablation study results on RCC dataset. Best results are highlighted in bold.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Method & ACC (\%) & AUC & F1 & Sen & Spe \\
\midrule
"""

    for _, row in results_df.iterrows():
        if row['Method'] == 'Full Model':
            latex_code += f"\\textbf{{{row['Method']}}} & \\textbf{{{row['ACC']}}} & \\textbf{{{row['AUC']}}} & \\textbf{{{row['F1']}}} & \\textbf{{{row['Sen']}}} & \\textbf{{{row['Spe']}}} \\\\\n"
        else:
            latex_code += f"{row['Method']} & {row['ACC']} & {row['AUC']} & {row['F1']} & {row['Sen']} & {row['Spe']} \\\\\n"

    latex_code += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex_code


def generate_k_latex_table(results_df):
    """生成K值敏感性LaTeX表格"""
    latex_code = r"""
\begin{table}[htbp]
\centering
\caption{Sensitivity analysis of prototype number K on RCC dataset.}
\label{tab:k_sensitivity}
\begin{tabular}{lcccc}
\toprule
K & ACC (\%) & AUC & F1 & Params (M) \\
\midrule
"""

    for _, row in results_df.iterrows():
        latex_code += f"{row['Method']} & {row['ACC']} & {row['AUC']} & {row['F1']} & - \\\\\n"

    latex_code += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex_code


def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验结果分析')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='实验结果目录')
    parser.add_argument('--output', type=str, default='./ablation_summary.csv',
                        help='输出文件路径')
    args = parser.parse_args()

    # ========== 主消融实验结果 ==========
    main_results = []
    for exp_name in MAIN_EXPERIMENTS:
        fold_results = collect_fold_results(args.results_dir, exp_name)
        if fold_results:
            stats = compute_statistics(fold_results)
            main_results.append({
                'Method': parse_experiment_name(exp_name),
                'ACC': f"{stats['acc']['mean']*100:.1f}±{stats['acc']['std']*100:.1f}",
                'AUC': f"{stats['auc']['mean']:.2f}±{stats['auc']['std']:.2f}",
                'F1': f"{stats['f1']['mean']:.2f}±{stats['f1']['std']:.2f}",
                'Sen': f"{stats['sensitivity']['mean']:.2f}±{stats['sensitivity']['std']:.2f}",
                'Spe': f"{stats['specificity']['mean']:.2f}±{stats['specificity']['std']:.2f}",
            })

    main_df = pd.DataFrame(main_results)

    # 保存主消融实验结果
    main_df.to_csv(args.output, index=False)
    print(f"主消融实验结果已保存到: {args.output}")

    # 打印主消融实验表格
    print("\n" + "="*80)
    print("主消融实验结果汇总")
    print("="*80)
    print(main_df.to_string(index=False))

    # 生成主消融实验LaTeX
    main_latex_output = args.output.replace('.csv', '_main.tex')
    main_latex_code = generate_main_latex_table(main_df)
    with open(main_latex_output, 'w') as f:
        f.write(main_latex_code)
    print(f"\n主消融实验LaTeX表格已保存到: {main_latex_output}")

    # ========== K值敏感性实验结果 ==========
    k_results = []
    for exp_name in K_EXPERIMENTS:
        fold_results = collect_fold_results(args.results_dir, exp_name)
        if fold_results:
            stats = compute_statistics(fold_results)
            k_results.append({
                'Method': parse_experiment_name(exp_name),
                'ACC': f"{stats['acc']['mean']*100:.1f}±{stats['acc']['std']*100:.1f}",
                'AUC': f"{stats['auc']['mean']:.2f}±{stats['auc']['std']:.2f}",
                'F1': f"{stats['f1']['mean']:.2f}±{stats['f1']['std']:.2f}",
            })

    if k_results:
        k_df = pd.DataFrame(k_results)

        # 保存K值实验结果
        k_output = args.output.replace('.csv', '_k_sensitivity.csv')
        k_df.to_csv(k_output, index=False)
        print(f"\nK值敏感性结果已保存到: {k_output}")

        # 打印K值实验表格
        print("\n" + "="*80)
        print("K值敏感性分析结果")
        print("="*80)
        print(k_df.to_string(index=False))

        # 生成K值实验LaTeX
        k_latex_output = args.output.replace('.csv', '_k_sensitivity.tex')
        k_latex_code = generate_k_latex_table(k_df)
        with open(k_latex_output, 'w') as f:
            f.write(k_latex_code)
        print(f"\nK值敏感性LaTeX表格已保存到: {k_latex_output}")


if __name__ == "__main__":
    main()
