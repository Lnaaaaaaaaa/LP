#!/bin/bash
# =============================================================================
# LP 批量实验后台离线训练脚本
# =============================================================================
# 用法: ./run_experiments.sh [模式]
#   模式:
#     all   - 训练全部任务 (默认)
#     k4    - 只训练 k=4 的任务
#     k10   - 只训练 k=10 的任务
#   示例:
#     ./run_experiments.sh        # 全部训练
#     ./run_experiments.sh k4     # 只训练 k=4
#     ./run_experiments.sh k10    # 只训练 k=10
#
# 支持断点续传：已完成的任务会自动跳过
# =============================================================================

# 解析命令行参数
MODE=${1:-all}

# 验证参数
if [[ "$MODE" != "all" && "$MODE" != "k4" && "$MODE" != "k10" ]]; then
    echo "错误: 无效的模式 '$MODE'"
    echo "用法: $0 [all|k4|k10]"
    echo "  all  - 训练全部任务 (默认)"
    echo "  k4   - 只训练 k=4 的任务"
    echo "  k10  - 只训练 k=10 的任务"
    exit 1
fi

# 日志目录
LOG_DIR="./experiment_logs"
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/train_${MODE}_${TIMESTAMP}.log"

echo "=========================================="
echo "LP 批量实验后台离线训练 (支持断点续传)"
echo "=========================================="
echo "训练模式: ${MODE}"
echo "启动时间: $(date)"
echo "主日志文件: ${MAIN_LOG}"
echo "=========================================="

# 启动后台训练
nohup bash -c '
    # 基础配置
    H5_FILE_DIR="/mnt/sda2/WSI/muti-modal/TCGA-RCC-fea/features"
    TEXT_MODEL_WEIGHTS="/mnt/sda1/ln_workspace/CONCH/checkpoints/pytorch_model.bin"
    INSTANCE_PROMPT="./text_prompt/TCGA_RCC_instance_prompt.json"
    BAG_PROMPT="./text_prompt/TCGA_RCC_bag_prompt.csv"
    NUM_CLASSES=3
    EPOCHS=20
    MODE="'"${MODE}"'"

    # 数据配置列表: (数据名, data_split_json, data_csv)
    declare -a DATA_CONFIGS=(
        "full|./data/data_split.json|./data/labels.csv"
        "1shot|./data_1shot/data_split.json|./data_1shot/labels.csv"
        "4shot|./data_4shot/data_split.json|./data_4shot/labels.csv"
        "16shot|./data_16shot/data_split.json|./data_16shot/labels.csv"
    )

    # K值列表
    declare -a K_VALUES=("4" "10")

    echo "=========================================="
    echo "LP 批量实验离线训练开始"
    echo "训练模式: ${MODE}"
    echo "=========================================="
    echo "开始时间: $(date)"
    echo ""

    # 定义训练任务函数
    run_task() {
        local task_num=$1
        local task_desc=$2
        local save_dir=$3
        local data_split_json=$4
        local data_csv=$5
        local K=$6

        # 根据模式过滤任务
        if [ "$MODE" = "k4" ] && [ "$K" != "4" ]; then
            return 0
        fi
        if [ "$MODE" = "k10" ] && [ "$K" != "10" ]; then
            return 0
        fi

        # 检查数据文件是否存在
        if [ ! -f "$data_split_json" ]; then
            echo "=========================================="
            echo "[${task_num}] ${task_desc} - 数据划分文件不存在，跳过"
            echo "=========================================="
            echo ""
            return 0
        fi

        if [ ! -f "$data_csv" ]; then
            echo "=========================================="
            echo "[${task_num}] ${task_desc} - 标签文件不存在，跳过"
            echo "=========================================="
            echo ""
            return 0
        fi

        # 检查任务是否已完成 (检查 summary_log.csv 是否存在)
        if [ -f "${save_dir}/logs/summary_log.csv" ]; then
            echo "=========================================="
            echo "[${task_num}] ${task_desc} - 已完成，跳过"
            echo "=========================================="
            echo ""
            return 0
        fi

        echo "=========================================="
        echo "[${task_num}] ${task_desc}"
        echo "=========================================="
        echo "开始时间: $(date)"
        echo "保存目录: ${save_dir}"

        python main.py \
            --data_split_json "$data_split_json" \
            --data_csv "$data_csv" \
            --h5_file_dir "$H5_FILE_DIR" \
            --instance_prompt "$INSTANCE_PROMPT" \
            --bag_prompt "$BAG_PROMPT" \
            --text_model_weights_path "$TEXT_MODEL_WEIGHTS" \
            --save_dir "$save_dir" \
            --K $K \
            --num_classes $NUM_CLASSES \
            --epochs $EPOCHS \
            --folds 5

        # 检查训练是否成功
        if [ -f "${save_dir}/logs/summary_log.csv" ]; then
            echo "完成时间: $(date)"
            echo ""
        else
            echo "错误: 训练失败，未生成 summary_log.csv"
            echo "完成时间: $(date)"
            echo ""
            return 1
        fi
    }

    # 训练任务列表
    task_id=1

    # ==================== full ====================
    run_task "${task_id}#full-k4" "full, k=4" "./results_gate/TCGA_RCC_full_k=4" "./data/data_split.json" "./data/labels.csv" "4"
    task_id=$((task_id + 1))
    run_task "${task_id}#full-k10" "full, k=10" "./results_gate/TCGA_RCC_full_k=10" "./data/data_split.json" "./data/labels.csv" "10"
    task_id=$((task_id + 1))

    # ==================== 1-shot ====================
    run_task "${task_id}#1shot-k4" "1-shot, k=4" "./results_gate/TCGA_RCC_1shot_k=4" "./data_1shot/data_split.json" "./data_1shot/labels.csv" "4"
    task_id=$((task_id + 1))
    run_task "${task_id}#1shot-k10" "1-shot, k=10" "./results_gate/TCGA_RCC_1shot_k=10" "./data_1shot/data_split.json" "./data_1shot/labels.csv" "10"
    task_id=$((task_id + 1))

    # ==================== 4-shot ====================
    run_task "${task_id}#4shot-k4" "4-shot, k=4" "./results_gate/TCGA_RCC_4shot_k=4" "./data_4shot/data_split.json" "./data_4shot/labels.csv" "4"
    task_id=$((task_id + 1))
    run_task "${task_id}#4shot-k10" "4-shot, k=10" "./results_gate/TCGA_RCC_4shot_k=10" "./data_4shot/data_split.json" "./data_4shot/labels.csv" "10"
    task_id=$((task_id + 1))

    # ==================== 16-shot ====================
    run_task "${task_id}#16shot-k4" "16-shot, k=4" "./results_gate/TCGA_RCC_16shot_k=4" "./data_16shot/data_split.json" "./data_16shot/labels.csv" "4"
    task_id=$((task_id + 1))
    run_task "${task_id}#16shot-k10" "16-shot, k=10" "./results_gate/TCGA_RCC_16shot_k=10" "./data_16shot/data_split.json" "./data_16shot/labels.csv" "10"

    echo "=========================================="
    echo "训练完成!"
    echo "结束时间: $(date)"
    echo "=========================================="

    # 汇总结果
    echo ""
    echo "=========================================="
    echo "结果汇总"
    echo "=========================================="

    for K in "4" "10"; do
        for DATA_NAME in "full" "1shot" "4shot" "16shot"; do
            RESULT_DIR="./results_gate/TCGA_RCC_${DATA_NAME}_k=${K}"
            SUMMARY_FILE="${RESULT_DIR}/logs/summary_log.csv"
            if [ -f "$SUMMARY_FILE" ]; then
                echo "--- TCGA_RCC_${DATA_NAME}_k=${K} ---"
                cat "$SUMMARY_FILE"
                echo ""
            fi
        done
    done
' > ${MAIN_LOG} 2>&1 &

PID=$!
echo ""
echo "后台进程已启动!"
echo "PID: ${PID}"
echo ""
echo "常用命令:"
echo "  查看日志: tail -f ${MAIN_LOG}"
echo "  检查进程: ps aux | grep ${PID}"
echo "  查看GPU:  watch -n 1 nvidia-smi"
echo ""
echo "断点续传说明:"
echo "  - 重新运行此脚本会自动跳过已完成的任务"
echo "  - 任务完成标志: results_gate/*/logs/summary_log.csv 存在"
echo ""
