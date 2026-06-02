#!/bin/bash
# =============================================================================
# LP 批量实验脚本
# =============================================================================
# 用法: bash scripts/run_experiments.sh
#
# 实验配置:
#   - K值: 4, 10
#   - 数据量: full, 1-shot, 4-shot, 16-shot
#   - 总计: 2 × 4 = 8 组实验
# =============================================================================

# 基础配置
H5_FILE_DIR="/mnt/sda2/WSI/muti-modal/TCGA-RCC-fea/features"
TEXT_MODEL_WEIGHTS="/mnt/sda1/ln_workspace/CONCH/checkpoints/pytorch_model.bin"
INSTANCE_PROMPT="./text_prompt/TCGA_RCC_instance_prompt.json"
BAG_PROMPT="./text_prompt/TCGA_RCC_bag_prompt.csv"
NUM_CLASSES=3
EPOCHS=20

# 数据配置列表: (数据名, data_split_json, data_csv)
declare -a DATA_CONFIGS=(
    "full|./data/data_split.json|./data/labels.csv"
    "1shot|./data_1shot/data_split.json|./data_1shot/labels.csv"
    "4shot|./data_4shot/data_split.json|./data_4shot/labels.csv"
    "16shot|./data_16shot/data_split.json|./data_16shot/labels.csv"
)

# K值列表
declare -a K_VALUES=("4" "10")

# 日志目录
LOG_DIR="./experiment_logs"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=============================================="
echo "LP 批量实验脚本"
echo "开始时间: $(date)"
echo "=============================================="

# 总实验计数
total_experiments=$((${#K_VALUES[@]} * ${#DATA_CONFIGS[@]}))
current_exp=0

# 遍历所有K值
for K in "${K_VALUES[@]}"; do
    # 遍历所有数据配置
    for DATA_CONFIG in "${DATA_CONFIGS[@]}"; do
        IFS='|' read -r DATA_NAME DATA_SPLIT DATA_CSV <<< "$DATA_CONFIG"

        current_exp=$((current_exp + 1))

        # 实验名称
        EXP_NAME="TCGA_RCC_${DATA_NAME}_k=${K}"
        SAVE_DIR="./results/${EXP_NAME}"
        LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}.log"

        echo ""
        echo "=============================================="
        echo "实验 [$current_exp/$total_experiments]: $EXP_NAME"
        echo "K=$K, 数据=$DATA_NAME"
        echo "保存目录: $SAVE_DIR"
        echo "日志文件: $LOG_FILE"
        echo "=============================================="

        # 检查数据文件是否存在
        if [ ! -f "$DATA_SPLIT" ]; then
            echo "[警告] 数据划分文件不存在: $DATA_SPLIT，跳过此实验"
            continue
        fi

        if [ ! -f "$DATA_CSV" ]; then
            echo "[警告] 标签文件不存在: $DATA_CSV，跳过此实验"
            continue
        fi

        # 运行实验
        echo "[信息] 开始训练..."

        python main.py \
            --data_split_json "$DATA_SPLIT" \
            --data_csv "$DATA_CSV" \
            --h5_file_dir "$H5_FILE_DIR" \
            --instance_prompt "$INSTANCE_PROMPT" \
            --bag_prompt "$BAG_PROMPT" \
            --text_model_weights_path "$TEXT_MODEL_WEIGHTS" \
            --save_dir "$SAVE_DIR" \
            --K $K \
            --num_classes $NUM_CLASSES \
            --epochs $EPOCHS \
            --folds 5 \
            2>&1 | tee "$LOG_FILE"

        # 检查实验结果
        if [ $? -eq 0 ]; then
            echo "[信息] 实验 $EXP_NAME 完成"
        else
            echo "[错误] 实验 $EXP_NAME 失败"
        fi
    done
done

echo ""
echo "=============================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
echo "日志目录: $LOG_DIR"
echo "=============================================="

# 汇总结果
echo ""
echo "=============================================="
echo "结果汇总"
echo "=============================================="

for K in "${K_VALUES[@]}"; do
    for DATA_CONFIG in "${DATA_CONFIGS[@]}"; do
        IFS='|' read -r DATA_NAME DATA_SPLIT DATA_CSV <<< "$DATA_CONFIG"
        EXP_NAME="TCGA_RCC_${DATA_NAME}_k=${K}"
        RESULT_DIR="./results/${EXP_NAME}"

        # 查找汇总结果文件
        SUMMARY_FILE="${RESULT_DIR}/logs/summary_log.csv"
        if [ -f "$SUMMARY_FILE" ]; then
            echo "--- $EXP_NAME ---"
            cat "$SUMMARY_FILE"
            echo ""
        fi
    done
done
