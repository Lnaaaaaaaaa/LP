#!/bin/bash
# 16-shot threshold 扫描实验
# 在已训练模型上测试不同threshold

# 配置
SEED=4
SHOTS=16
BASE_DIR="/mnt/sda1/ln_workspace/LP/results/camelyon16/16shot_threshold_scan"

# 要测试的threshold值
THRESHOLDS=(0.25 0.30 0.35 0.40 0.45 0.50)

echo "=========================================="
echo "16-shot Threshold 扫描实验"
echo "=========================================="

for THRESH in "${THRESHOLDS[@]}"; do
    echo ""
    echo ">>> Testing threshold = $THRESH"

    SAVE_DIR="${BASE_DIR}/threshold_${THRESH}_seed${SEED}"

    python main.py \
        --data_split_json /mnt/sda1/ln_workspace/Libra-MIL/data/DATA-CAMELYON16/data_16shot/data_split.json \
        --data_csv /mnt/sda1/ln_workspace/Libra-MIL/data/DATA-CAMELYON16/process_camelyon16.csv \
        --h5_file_dir /mnt/sda1/ln_workspace/Libra-MIL/data/DATA-CAMELYON16/h5_files \
        --instance_prompt /mnt/sda1/ln_workspace/Libra-MIL/data/DATA-CAMELYON16/instance_prompts.json \
        --bag_prompt /mnt/sda1/ln_workspace/Libra-MIL/data/DATA-CAMELYON16/bag_prompts.csv \
        --save_dir "$SAVE_DIR" \
        --K 10 \
        --K_t 29 \
        --num_classes 3 \
        --threshold $THRESH \
        --seed $SEED \
        --epochs 50 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --patience 15 \
        --tau_init 1.0 \
        --tau_min 0.05 \
        --tau_decay_rate 0.95 \
        --ema_momentum 0.9 \
        --alpha_ptc 0.1 \
        --num_heads 8 \
        --ot_epsilon 0.05 \
        --ot_iters 20

    echo ">>> Threshold $THRESH done, results saved to $SAVE_DIR"
done

echo ""
echo "=========================================="
echo "所有实验完成！结果汇总："
echo "=========================================="

# 汇总结果
for THRESH in "${THRESHOLDS[@]}"; do
    RESULT_FILE="${BASE_DIR}/threshold_${THRESH}_seed${SEED}/logs/summary_log.csv"
    if [ -f "$RESULT_FILE" ]; then
        echo "Threshold $THRESH:"
        cat "$RESULT_FILE"
        echo ""
    fi
done
