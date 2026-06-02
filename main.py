#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LP 训练主脚本
=============

用法:
    python main.py --config configs/default.yaml

或使用命令行参数:
    python main.py \\
        --data_split_json ./data/tcga_split.json \\
        --data_csv ./data/labels.csv \\
        --h5_file_dir ./data/features/ \\
        --instance_prompt ./text_prompt/TCGA_RCC_instance_prompt.json \\
        --bag_prompt ./text_prompt/TCGA_RCC_bag_prompt.csv \\
        --text_model_weights_path ./conch/pytorch_model.bin \\
        --save_dir ./results/ \\
        --K 4 \\
        --num_classes 2
"""

import os
import sys
import argparse
import json

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import LPModel, TextEncoder
from models.text_encoder import load_text_prompts
from datasets import get_dataloader
from utils import (
    compute_metrics,
    CSVWriter,
    set_seed,
    EarlyStopping,
    TemperatureScheduler,
)


def parse_args():
    parser = argparse.ArgumentParser(description='LP: Libra-PTCMIL Training')

    parser.add_argument('--data_split_json', type=str, required=True)
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--h5_file_dir', type=str, required=True)
    parser.add_argument('--instance_prompt', type=str, required=True)
    parser.add_argument('--bag_prompt', type=str, required=True)

    parser.add_argument('--text_model_weights_path', type=str, default='conch/pytorch_model.bin')
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--K_t', type=int, default=46)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)

    parser.add_argument('--tau_init', type=float, default=1.0)
    parser.add_argument('--tau_min', type=float, default=0.05)
    parser.add_argument('--tau_decay_rate', type=float, default=0.95)

    parser.add_argument('--ema_momentum', type=float, default=0.9)
    parser.add_argument('--alpha_ptc', type=float, default=0.1)
    parser.add_argument('--ot_epsilon', type=float, default=0.05)
    parser.add_argument('--ot_iters', type=int, default=20)

    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


def train_one_epoch(model, train_loader, optimizer, temp_scheduler, device, epoch, num_classes):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []

    tau = temp_scheduler.get_tau(epoch)
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train] (tau={tau:.3f})')

    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.squeeze(0).to(device)
        labels = labels.to(device)

        output = model(features, labels=labels, tau=tau)
        loss = output['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.update_ema()

        total_loss += loss.item()
        probs = torch.softmax(output['logits'], dim=-1)
        pred = torch.argmax(probs, dim=-1)

        all_labels.append(labels.item())
        all_preds.append(pred.item())
        all_scores.append(probs.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    acc, auc, f1 = compute_metrics(all_labels, all_preds, all_scores, num_classes=num_classes)

    return {'loss': total_loss / len(train_loader), 'acc': acc, 'auc': auc, 'f1': f1}


def train_one_fold(args, fold, text_encoder, P_text, prompt_bag, device):
    """训练单个fold"""
    print(f"\n{'='*60}")
    print(f"Fold {fold}")
    print(f"{'='*60}")

    # 设置随机种子
    set_seed(args.seed)

    # 创建保存目录
    fold_save_dir = os.path.join(args.save_dir, 'checkpoints', f'best_model_{fold}.pt')
    os.makedirs(os.path.dirname(fold_save_dir), exist_ok=True)

    # 加载数据
    print(f"\n[信息] 加载数据...")
    dataloaders = get_dataloader(
        data_split_json=args.data_split_json,
        data_csv=args.data_csv,
        h5_file_dir=args.h5_file_dir,
        idx=fold,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # 创建模型
    print("\n[信息] 创建模型...")
    model = LPModel(
        dim=args.dim,
        K=args.K,
        K_t=args.K_t,
        num_classes=args.num_classes,
        P_text=P_text,
        prompt_bag=prompt_bag,
        tau_init=args.tau_init,
        tau_min=args.tau_min,
        ema_momentum=args.ema_momentum,
        alpha_ptc=args.alpha_ptc,
        ot_epsilon=args.ot_epsilon,
        ot_iters=args.ot_iters,
        num_heads=args.num_heads,
    ).to(device)

    print(f"[信息] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    temp_scheduler = TemperatureScheduler(
        tau_init=args.tau_init,
        tau_min=args.tau_min,
        decay_rate=args.tau_decay_rate
    )
    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    # 创建日志写入器
    log_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_writer = CSVWriter(
        os.path.join(log_dir, f'log_{fold}.csv'),
        header=['epoch', 'train_loss', 'train_acc', 'train_auc', 'train_f1',
                'val_loss', 'val_acc', 'val_auc', 'val_f1', 'tau']
    )

    # 训练
    print("\n[信息] 开始训练...")
    best_val_auc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model, dataloaders['train'], optimizer, temp_scheduler, device, epoch, args.num_classes
        )

        # 验证
        model.eval()
        val_loss = 0.0
        val_labels, val_preds, val_scores = [], [], []
        with torch.no_grad():
            for features, labels in dataloaders['valid']:
                features = features.squeeze(0).to(device)
                labels = labels.to(device)
                output = model(features, labels=labels, tau=args.tau_min)
                val_loss += output['loss'].item()
                probs = torch.softmax(output['logits'], dim=-1)
                pred = torch.argmax(probs, dim=-1)
                val_labels.append(labels.item())
                val_preds.append(pred.item())
                val_scores.append(probs.cpu().numpy())

        val_acc, val_auc, val_f1 = compute_metrics(val_labels, val_preds, val_scores, num_classes=args.num_classes)

        scheduler.step()
        tau = temp_scheduler.get_tau(epoch)

        log_writer.write_row([
            epoch,
            f"{train_metrics['loss']:.4f}",
            f"{train_metrics['acc']:.4f}",
            f"{train_metrics['auc']:.4f}",
            f"{train_metrics['f1']:.4f}",
            f"{val_loss / len(dataloaders['valid']):.4f}",
            f"{val_acc:.4f}",
            f"{val_auc:.4f}",
            f"{val_f1:.4f}",
            f"{tau:.4f}",
        ])

        print(f"Epoch {epoch:3d} | Train AUC: {train_metrics['auc']:.4f} | Val AUC: {val_auc:.4f} | tau: {tau:.3f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), fold_save_dir)

        if early_stopping(val_auc):
            print(f"\n[信息] 早停触发，在 epoch {epoch}")
            break

    print(f"\n[信息] Fold {fold} 训练完成，最佳验证 AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    # 加载最佳模型并测试
    model.load_state_dict(torch.load(fold_save_dir, weights_only=True))
    model.eval()
    test_labels, test_preds, test_scores = [], [], []
    with torch.no_grad():
        for features, labels in dataloaders['test']:
            features = features.squeeze(0).to(device)
            labels = labels.to(device)
            output = model(features, tau=args.tau_min)
            probs = torch.softmax(output['logits'], dim=-1)
            pred = torch.argmax(probs, dim=-1)
            test_labels.append(labels.item())
            test_preds.append(pred.item())
            test_scores.append(probs.cpu().numpy())

    test_acc, test_auc, test_f1 = compute_metrics(test_labels, test_preds, test_scores, num_classes=args.num_classes)
    print(f"[Fold {fold} 测试结果] ACC: {test_acc:.4f} | AUC: {test_auc:.4f} | F1: {test_f1:.4f}")

    return {
        'fold': fold,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # 创建保存目录
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)

    # 保存配置
    config_dict = vars(args)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[信息] 使用设备: {device}")

    # 加载文本原型（所有fold共享）
    print("\n[信息] 加载文本原型...")
    text_encoder = TextEncoder(weights_path=args.text_model_weights_path)
    P_text, prompt_bag = load_text_prompts(
        args.instance_prompt,
        args.bag_prompt,
        text_encoder,
        device=device
    )

    # 创建最终结果日志
    final_log = CSVWriter(
        os.path.join(args.save_dir, 'logs', 'final_log.csv'),
        header=['fold', 'test_acc', 'test_auc', 'test_f1']
    )

    # K折交叉验证
    all_results = []
    for fold in range(args.folds):
        result = train_one_fold(args, fold, text_encoder, P_text, prompt_bag, device)
        all_results.append(result)
        final_log.write_row([fold, result['test_acc'], result['test_auc'], result['test_f1']])

    # 计算汇总统计
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")

    accs = [r['test_acc'] for r in all_results]
    aucs = [r['test_auc'] for r in all_results]
    f1s = [r['test_f1'] for r in all_results]

    summary = {
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs),
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'mean_f1': np.mean(f1s),
        'std_f1': np.std(f1s),
    }

    print(f"ACC: {summary['mean_acc']:.4f} +/- {summary['std_acc']:.4f}")
    print(f"AUC: {summary['mean_auc']:.4f} +/- {summary['std_auc']:.4f}")
    print(f"F1:  {summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f}")

    # 保存汇总结果
    summary_log = CSVWriter(
        os.path.join(args.save_dir, 'logs', 'summary_log.csv'),
        header=['metric', 'test_acc', 'test_auc', 'test_f1']
    )
    summary_log.write_row(['mean', summary['mean_acc'], summary['mean_auc'], summary['mean_f1']])
    summary_log.write_row(['std', summary['std_acc'], summary['std_auc'], summary['std_f1']])

    print(f"\n[信息] 所有结果已保存到: {args.save_dir}")


if __name__ == '__main__':
    main()
