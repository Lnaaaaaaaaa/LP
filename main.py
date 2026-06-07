"""
LP: 多模态原型立体融合模型训练脚本
===================================
用于少样本学习任务的全切片图像(WSI)分类

核心流程:
    1. 加载并编码文本提示（使用CONCH模型）
    2. K折交叉验证训练
    3. 保存最佳模型和训练日志
"""

# ==================== 导入依赖库 ====================
import os
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm  # 进度条显示

# 项目内部模块
from models import LPModel, TextEncoder  # 主模型和文本编码器
from datasets import get_dataloader       # 数据加载器
from utils import (
    compute_metrics,       # 计算ACC/AUC/F1指标
    CSVWriter,             # CSV日志写入器
    write_summary_log,     # 汇总多折结果
    set_seed,              # 设置随机种子
    EarlyStopping,         # 早停机制
    TemperatureScheduler,  # 温度退火调度器
)


# ==================== 命令行参数配置 ====================
def get_config():
    """
    解析命令行参数，配置训练超参数和路径设置

    Returns:
        args: 包含所有配置参数的命名空间对象
    """
    import argparse
    parser = argparse.ArgumentParser(description='LP: Libra-PTCMIL Training')

    # ---------- 数据路径配置 ----------
    parser.add_argument('--data_split_json', type=str, required=True,
                        help='数据集划分JSON文件路径，包含train_0/val_0/test_0等')
    parser.add_argument('--data_csv', type=str, required=True,
                        help='数据集CSV文件路径，包含name和label两列')
    parser.add_argument('--h5_file_dir', type=str, required=True,
                        help='WSI patch特征的h5文件目录')
    parser.add_argument('--instance_prompt', type=str, required=True,
                        help='实例级文本提示JSON文件路径，用于最优传输对齐')
    parser.add_argument('--bag_prompt', type=str, required=True,
                        help='Bag级文本提示CSV文件路径，用于最终分类')
    parser.add_argument('--save_dir', type=str, default='./results/',
                        help='模型和日志保存目录')
    parser.add_argument('--text_model_weights_path', type=str, default='conch/pytorch_model.bin',
                        help='CONCH文本模型权重路径')

    # ---------- 模型超参数 ----------
    parser.add_argument('--dim', type=int, default=512,
                        help='特征维度（需与预提取特征维度一致）')
    parser.add_argument('--K', type=int, default=4,
                        help='视觉原型数量，用于聚类patch特征')
    parser.add_argument('--K_t', type=int, default=46,
                        help='文本原型数量，由instance_prompt文件决定')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='分类类别数')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='交叉注意力的头数')
    parser.add_argument('--ot_epsilon', type=float, default=0.05,
                        help='Sinkhorn最优传输的熵正则化系数')
    parser.add_argument('--ot_iters', type=int, default=20,
                        help='Sinkhorn最优传输的迭代次数')

    # ---------- 训练超参数 ----------
    parser.add_argument('--folds', type=int, default=5,
                        help='K折交叉验证的折数')
    parser.add_argument('--epochs', type=int, default=50,
                        help='每折训练的轮数（epoch数）')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减（L2正则化）')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值，验证AUC连续N轮不提升则停止')
    parser.add_argument('--seed', type=int, default=7,
                        help='随机种子，确保实验可复现')

    # ---------- 温度调度参数 ----------
    # 温度tau控制软分配的"软硬程度"：
    #   - tau大：分配更平滑，梯度流畅
    #   - tau小：分配更接近硬聚类，边界清晰
    # 训练过程中tau从tau_init逐渐衰减到tau_min
    parser.add_argument('--tau_init', type=float, default=1.0, help='初始温度')
    parser.add_argument('--tau_min', type=float, default=0.05, help='最小温度')
    parser.add_argument('--tau_decay_rate', type=float, default=0.95, help='温度衰减率')

    # ---------- 其他参数 ----------
    parser.add_argument('--ema_momentum', type=float, default=0.9,
                        help='视觉原型EMA更新的动量')
    parser.add_argument('--alpha_ptc', type=float, default=0.1,
                        help='PTC正交损失权重，鼓励原型相互正交')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的并行进程数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备：cuda或cpu')

    return parser.parse_args()


# ==================== 训练函数 ====================
def train(model, dataloader, optimizer, temp_scheduler, device, epoch, num_classes):
    """
    单轮训练过程（一个epoch）

    流程:
        1. 前向传播计算损失
        2. 反向传播更新参数
        3. EMA更新视觉原型
        4. 收集预测结果用于计算指标

    Args:
        model: LP模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        temp_scheduler: 温度调度器
        device: 计算设备
        epoch: 当前轮次（用于获取温度值）
        num_classes: 类别数

    Returns:
        tuple: (平均损失, 准确率, AUC, F1分数)
    """
    model.train()  # 设置为训练模式
    total_loss = 0.0
    y_true, y_pred, y_score = [], [], []

    # 获取当前温度值（温度随epoch衰减）
    tau = temp_scheduler.get_tau(epoch)
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train] (tau={tau:.3f})")

    for feats, labels in loop:
        # 数据移动到设备
        # feats形状: (1, N, D)，因为batch_size=1，N是该WSI的patch数
        feats = feats.squeeze(0).to(device)  # (N, D)
        labels = labels.to(device)

        # 前向传播
        output = model(feats, labels=labels, tau=tau)
        loss = output['loss']

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数

        # EMA更新视觉原型
        # 跨WSI稳定原型，防止batch_size=1导致的震荡
        model.update_ema()

        # 收集预测结果
        total_loss += loss.item()
        probs = torch.softmax(output['logits'], dim=-1)
        preds = torch.argmax(probs, dim=-1)

        y_true.append(labels.item())
        y_pred.append(preds.item())
        y_score.append(probs.detach().cpu().numpy())

        loop.set_postfix({'loss': f'{loss.item():.4f}'})

    # 计算本轮指标
    acc, auc, f1 = compute_metrics(y_true, y_pred, y_score, num_classes)
    return total_loss / len(dataloader), acc, auc, f1


# ==================== 评估函数 ====================
def evaluate(model, dataloader, device, epoch, num_classes, tau=None):
    """
    模型评估过程（验证集/测试集）

    与训练的区别:
        - 使用torch.no_grad()禁用梯度计算
        - 不进行反向传播和参数更新
        - 不进行EMA更新

    Args:
        model: LP模型
        dataloader: 数据加载器
        device: 计算设备
        epoch: 当前轮次
        num_classes: 类别数
        tau: 温度值，验证/测试时使用最小温度

    Returns:
        tuple: (准确率, AUC, F1分数)
    """
    model.eval()  # 设置为评估模式
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():  # 禁用梯度计算，节省内存
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1} [Eval]")
        for feats, labels in loop:
            feats = feats.squeeze(0).to(device)
            labels = labels.to(device)

            # 前向传播（不计算损失，只获取logits）
            output = model(feats, labels=labels, tau=tau)
            logits = output['logits']

            # 收集预测结果
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            y_true.append(labels.item())
            y_pred.append(preds.item())
            y_score.append(probs.cpu().numpy())

    y_score = np.array(y_score)
    return compute_metrics(y_true, y_pred, y_score, num_classes)


# ==================== 主函数 ====================
def main():
    """
    主训练流程：
    1. 加载配置和初始化环境
    2. 加载并编码文本提示（使用CONCH模型）
    3. K折交叉验证训练
    4. 保存模型和日志
    """
    # ---------- 初始化配置 ----------
    args = get_config()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)  # 设置随机种子，确保可复现

    print(f"[信息] 使用设备: {device}")

    # ---------- 创建输出目录 ----------
    # checkpoints: 保存模型权重（best_model_0.pt等）
    # logs: 保存训练日志（log_0.csv、final_log.csv等）
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)

    # ---------- 加载文本提示 ----------
    # 文本提示分为两个层级：
    #   - 实例级(instance_prompt)：用于最优传输对齐，将patch特征与文本原型匹配
    #   - Bag级(bag_prompt)：用于最终分类，作为类别原型
    print("\n[信息] 加载文本原型...")

    # 加载实例级文本提示（JSON格式）
    # 格式: {"prompt_0": "文本描述0", "prompt_1": "文本描述1", ...}
    # 或: ["文本描述0", "文本描述1", ...]
    with open(args.instance_prompt, 'r', encoding='utf-8') as f:
        instance_prompts = json.load(f)

    # 加载Bag级文本提示（CSV格式）
    # 格式: 每行一个类别的文本描述
    bag_prompts_df = pd.read_csv(args.bag_prompt)

    # ---------- 编码文本提示 ----------
    # 使用CONCH模型将文本编码为特征向量
    # CONCH是病理学专用的视觉-语言模型，理解医学术语
    text_encoder = TextEncoder(weights_path=args.text_model_weights_path).to(device)
    text_encoder.eval()

    with torch.no_grad():
        # 实例级文本原型编码
        # P_text形状: (K_t, D)，K_t是文本原型数量，D是特征维度(512)
        if isinstance(instance_prompts, dict):
            # 如果是字典，取所有值
            instance_texts = list(instance_prompts.values())
        else:
            # 如果是列表，直接使用
            instance_texts = instance_prompts
        P_text = text_encoder(instance_texts)

        # Bag级文本原型编码
        # prompt_bag形状: (num_classes, D)
        if 'prompt' in bag_prompts_df.columns:
            bag_texts = bag_prompts_df['prompt'].tolist()
        else:
            # 如果没有'prompt'列，取第一列
            bag_texts = bag_prompts_df.iloc[:, 0].tolist()
        prompt_bag = text_encoder(bag_texts)

    # 更新K_t为实际加载的文本原型数量
    # 因为实际数量可能由文件内容决定，而非命令行参数
    args.K_t = P_text.shape[0]

    # 创建最终结果日志（记录每折的测试结果）
    final_csv = CSVWriter(
        filename=os.path.join(args.save_dir, 'logs', 'final_log.csv'),
        header=['fold', 'test_acc', 'test_auc', 'test_f1']
    )

    # ---------- K折交叉验证训练 ----------
    # K折交叉验证：将数据分成K份，每次用K-1份训练，1份测试
    # 最终取K次结果的平均值，减少数据划分的偶然性
    for fold in range(args.folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")

        # 每个fold重新设置种子
        # 使用 seed + fold 确保每个fold的shuffle顺序不同但可复现
        set_seed(args.seed + fold)

        # 初始化模型
        # P_text和prompt_bag是预编码的文本原型，作为模型的固定组件
        model = LPModel(
            dim=args.dim,
            K=args.K,
            K_t=args.K_t,
            num_classes=args.num_classes,
            P_text=P_text,           # 实例级文本原型
            prompt_bag=prompt_bag,    # Bag级文本原型
            tau_init=args.tau_init,
            tau_min=args.tau_min,
            ema_momentum=args.ema_momentum,
            alpha_ptc=args.alpha_ptc,
            ot_epsilon=args.ot_epsilon,
            ot_iters=args.ot_iters,
            num_heads=args.num_heads,
        ).to(device)

        print(f"[信息] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 获取数据加载器
        # 返回 {'train': ..., 'valid': ..., 'test': ...}
        # generator参数确保shuffle顺序固定
        loaders = get_dataloader(
            data_split_json=args.data_split_json,
            data_csv=args.data_csv,
            h5_file_dir=args.h5_file_dir,
            idx=fold,  # 当前折索引
            num_workers=args.num_workers,
            seed=args.seed
        )

        # 配置优化器和学习率调度器
        # AdamW: 带权重衰减的Adam优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # CosineAnnealingLR: 余弦退火学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        # 温度调度器：控制软分配的"软硬程度"
        temp_scheduler = TemperatureScheduler(
            tau_init=args.tau_init,
            tau_min=args.tau_min,
            decay_rate=args.tau_decay_rate
        )
        # 早停机制：验证AUC连续patience轮不提升则停止
        early_stopping = EarlyStopping(patience=args.patience, mode='max')

        # 训练状态变量
        best_auc = 0.0      # 最佳验证AUC
        best_epoch = 0      # 最佳epoch编号

        # 创建每折训练日志
        log_csv = CSVWriter(
            filename=os.path.join(args.save_dir, 'logs', f'log_{fold}.csv'),
            header=['epoch', 'train_loss', 'train_acc', 'train_auc', 'train_f1',
                    'valid_acc', 'valid_auc', 'valid_f1', 'tau']
        )

        # ---------- 单折训练循环 ----------
        for epoch in range(args.epochs):
            # 训练一个epoch
            train_loss, train_acc, train_auc, train_f1 = train(
                model, loaders['train'], optimizer, temp_scheduler, device, epoch, args.num_classes
            )

            # 验证
            # 验证时使用最小温度tau_min，使分配更接近硬聚类
            valid_acc, valid_auc, valid_f1 = evaluate(
                model, loaders['valid'], device, epoch, args.num_classes, tau=args.tau_min
            )

            # 获取当前温度并更新学习率
            tau = temp_scheduler.get_tau(epoch)
            scheduler.step()

            # 记录本轮结果到日志
            log_csv.write_row([
                epoch + 1,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{train_auc:.4f}",
                f"{train_f1:.4f}",
                f"{valid_acc:.4f}",
                f"{valid_auc:.4f}",
                f"{valid_f1:.4f}",
                f"{tau:.4f}",
            ])

            print(f"Epoch {epoch+1:3d} | Train AUC: {train_auc:.4f} | Val AUC: {valid_auc:.4f} | tau: {tau:.3f}")

            # 模型保存与早停判断
            # 使用验证集AUC作为指标，越大越好
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_epoch = epoch + 1
                # 保存最佳模型权重
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'checkpoints/best_model_{fold}.pt'))

            # 早停检查
            if early_stopping(valid_auc):
                print(f"\n[信息] 早停触发，在 epoch {epoch+1}")
                break

        print(f"\n[信息] Fold {fold} 训练完成，最佳验证 AUC: {best_auc:.4f} (epoch {best_epoch})")

        # ---------- 加载最佳模型并测试 ----------
        # 训练结束后，加载验证集上表现最好的模型进行测试
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f'checkpoints/best_model_{fold}.pt'), weights_only=True))
        test_acc, test_auc, test_f1 = evaluate(model, loaders['test'], device, best_epoch, args.num_classes, tau=args.tau_min)
        print(f"[Fold {fold} 测试结果] ACC: {test_acc:.4f} | AUC: {test_auc:.4f} | F1: {test_f1:.4f}")

        # 记录到最终结果日志
        final_csv.write_row([fold, test_acc, test_auc, test_f1])

    # ---------- 输出汇总结果 ----------
    # 计算所有fold的均值和标准差
    write_summary_log(
        os.path.join(args.save_dir, 'logs', 'final_log.csv'),
        os.path.join(args.save_dir, 'logs', 'summary_log.csv')
    )

    print(f"\n[信息] 所有结果已保存到: {args.save_dir}")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()
