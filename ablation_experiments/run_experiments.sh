#!/bin/bash
# 消融实验批量运行脚本
# ======================
# 用于BIBM 2026投稿

set -e  # 遇到错误立即退出

# 配置路径
CONFIG_FILE="./config_rcc.yaml"
OUTPUT_DIR="./results"
LOG_DIR="./logs"

# 创建日志目录
mkdir -p ${LOG_DIR}

# 主消融实验列表
MAIN_EXPERIMENTS=(
    "Full"
    "A1_no_dynamic_prototype"
    "A2_no_ot_fusion"
    "A3_no_gating"
    "A4_no_bag_prompt"
)

# K值敏感性实验列表
K_EXPERIMENTS=(
    "A5_K2"
    "A5_K4"
    "A5_K6"
    "A5_K8"
    "A5_K16"
)

echo "=========================================="
echo "消融实验批量运行"
echo "=========================================="
echo "配置文件: ${CONFIG_FILE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "主消融实验数量: ${#MAIN_EXPERIMENTS[@]}"
echo "K值实验数量: ${#K_EXPERIMENTS[@]}"
echo "=========================================="

# 运行主消融实验
echo ""
echo "========== 主消融实验 =========="
for exp in "${MAIN_EXPERIMENTS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "运行实验: ${exp}"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------"

    LOG_FILE="${LOG_DIR}/${exp}.log"

    python run_ablation.py \
        --config ${CONFIG_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --experiments ${exp} \
        2>&1 | tee ${LOG_FILE}

    echo "实验 ${exp} 完成，日志保存到: ${LOG_FILE}"
done

# 运行K值敏感性实验
echo ""
echo "========== K值敏感性分析 =========="
for exp in "${K_EXPERIMENTS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "运行实验: ${exp}"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------"

    LOG_FILE="${LOG_DIR}/${exp}.log"

    python run_ablation.py \
        --config ${CONFIG_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --experiments ${exp} \
        2>&1 | tee ${LOG_FILE}

    echo "实验 ${exp} 完成，日志保存到: ${LOG_FILE}"
done

echo ""
echo "=========================================="
echo "所有实验完成!"
echo "=========================================="
echo "正在分析结果..."

# 运行结果分析
python analyze_results.py --results_dir ${OUTPUT_DIR}

echo "完成! 结果已保存到 ablation_summary.csv"
