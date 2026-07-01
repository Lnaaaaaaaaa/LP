#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验自动化运行脚本
======================
用于BIBM 2026投稿，RCC数据集上的消融实验

使用方法:
    python run_ablation.py --config config_rcc.yaml

功能:
    1. 自动运行所有消融实验配置
    2. 收集所有实验结果
    3. 生成消融实验表格
"""

import os
import json
import yaml
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path


# ==================== 消融实验配置 ====================
ABLATION_CONFIGS = {
    # 完整模型 (Baseline)
    "Full": {
        "description": "Full Model",
        "params": {
            "K": 4,
            "use_dynamic_prototype": True,
            "use_ot_fusion": True,
            "use_gating": True,
            "use_bag_prompt": True,
        }
    },

    # A1: 动态原型消融
    "A1_no_dynamic_prototype": {
        "description": "w/o Dynamic Prototype",
        "params": {
            "use_dynamic_prototype": False,
        }
    },

    # A2: 最优传输消融 (OT → 相似度)
    "A2_no_ot_fusion": {
        "description": "w/o Optimal Transport",
        "params": {
            "use_ot_fusion": False,
        }
    },

    # A3: 门控机制消融
    "A3_no_gating": {
        "description": "w/o Gating Mechanism",
        "params": {
            "use_gating": False,
        }
    },

    # A4: 包级先验知识消融
    "A4_no_bag_prompt": {
        "description": "w/o Bag-level Prior",
        "params": {
            "use_bag_prompt": False,
        }
    },

    # A5: K值敏感性分析
    "A5_K2": {
        "description": "K=2",
        "params": {
            "K": 2,
        }
    },
    "A5_K4": {
        "description": "K=4 (Baseline)",
        "params": {
            "K": 4,
        }
    },
    "A5_K6": {
        "description": "K=6",
        "params": {
            "K": 6,
        }
    },
    "A5_K8": {
        "description": "K=8",
        "params": {
            "K": 8,
        }
    },
    "A5_K16": {
        "description": "K=16",
        "params": {
            "K": 16,
        }
    },
}


def load_base_config(config_path):
    """加载基础配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_command(base_config, ablation_params, exp_name, output_dir):
    """构建训练命令"""
    cmd = [
        "python", "main.py",
        "--data_split_json", base_config['data_split_json'],
        "--data_csv", base_config['data_csv'],
        "--h5_file_dir", base_config['h5_file_dir'],
        "--instance_prompt", base_config['instance_prompt'],
        "--bag_prompt", base_config['bag_prompt'],
        "--text_model_weights_path", base_config['text_model_weights_path'],
        "--save_dir", str(output_dir / exp_name),
        "--dim", str(base_config.get('dim', 512)),
        "--K", str(base_config.get('K', 4)),
        "--K_t", str(base_config.get('K_t', 33)),
        "--num_classes", str(base_config.get('num_classes', 3)),
        "--num_heads", str(base_config.get('num_heads', 8)),
        "--ot_epsilon", str(base_config.get('ot_epsilon', 0.05)),
        "--ot_iters", str(base_config.get('ot_iters', 20)),
        "--folds", str(base_config.get('folds', 5)),
        "--epochs", str(base_config.get('epochs', 50)),
        "--lr", str(base_config.get('lr', 1e-4)),
        "--weight_decay", str(base_config.get('weight_decay', 1e-4)),
        "--patience", str(base_config.get('patience', 15)),
        "--seed", str(base_config.get('seed', 4)),
        "--threshold", str(base_config.get('threshold', 0.5)),
    ]

    # 添加消融参数
    for key, value in ablation_params.items():
        if isinstance(value, bool):
            cmd.extend([f"--{key}", str(value).lower()])
        else:
            cmd.extend([f"--{key}", str(value)])

    return cmd


def run_experiment(cmd, exp_name):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"运行实验: {exp_name}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")

    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds() / 60

    return {
        "exp_name": exp_name,
        "duration_min": duration,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def collect_results(output_dir):
    """收集所有实验结果"""
    results = []

    for exp_name in ABLATION_CONFIGS.keys():
        summary_file = output_dir / exp_name / "summary_log.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            # 计算均值和标准差
            acc_mean = df['test_acc'].mean()
            acc_std = df['test_acc'].std()
            auc_mean = df['test_auc'].mean()
            auc_std = df['test_auc'].std()
            f1_mean = df['test_f1'].mean()
            f1_std = df['test_f1'].std()

            results.append({
                "Method": ABLATION_CONFIGS[exp_name]["description"],
                "ACC (%)": f"{acc_mean*100:.1f}±{acc_std*100:.1f}",
                "AUC": f"{auc_mean:.2f}±{auc_std:.2f}",
                "F1": f"{f1_mean:.2f}±{f1_std:.2f}",
            })

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验自动化运行')
    parser.add_argument('--config', type=str, required=True,
                        help='基础配置文件路径 (YAML格式)')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                        help='结果输出目录')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='要运行的实验名称，默认运行全部')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印命令，不实际运行')
    args = parser.parse_args()

    # 加载基础配置
    base_config = load_base_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选择要运行的实验
    exp_names = args.experiments if args.experiments else list(ABLATION_CONFIGS.keys())

    # 运行实验
    for exp_name in exp_names:
        if exp_name not in ABLATION_CONFIGS:
            print(f"警告: 未知实验名称 {exp_name}，跳过")
            continue

        ablation_config = ABLATION_CONFIGS[exp_name]
        cmd = build_command(base_config, ablation_config["params"], exp_name, output_dir)

        if args.dry_run:
            print(f"\n实验: {exp_name}")
            print(f"描述: {ablation_config['description']}")
            print(f"命令: {' '.join(cmd)}")
        else:
            result = run_experiment(cmd, exp_name)
            if result["returncode"] != 0:
                print(f"实验 {exp_name} 失败!")
                print(f"错误: {result['stderr']}")

    # 收集结果
    if not args.dry_run:
        results_df = collect_results(output_dir)
        results_df.to_csv(output_dir / "ablation_results.csv", index=False)
        print("\n" + "="*60)
        print("消融实验结果汇总")
        print("="*60)
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
