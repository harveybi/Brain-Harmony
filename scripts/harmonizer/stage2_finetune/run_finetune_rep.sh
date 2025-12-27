#!/bin/bash

# 检查参数
if [ $# -ne 6 ]; then
    echo "用法: $0 <模型大小> <num_latent_tokens> <数据集> <随机种子> <数据根目录> <输出目录>"
    echo "模型大小: base, small, large"
    echo "数据集: 数据集名称"
    echo "随机种子: 用于数据分割的随机种子"
    echo "数据根目录: 例如 /p/project1/hai_1024/data"
    echo "输出目录: 例如 /p/project1/hai_1024/Brain-Harmony/experiments"
    echo "例如: $0 base 128 ADNI 42 /p/project1/hai_1024/data /p/project1/hai_1024/Brain-Harmony/experiments"
    exit 1
fi

# 获取参数
MODEL_SIZE=$1
NUM_LATENT_TOKENS=$2
DATASET_NAME=$3
SPLIT_SEED=$4
DATA_ROOT=$5
OUTPUT_ROOT=$6
NUM_WORKERS="${NUM_WORKERS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# 验证模型大小并设置对应的缩写
case $MODEL_SIZE in
    "base")
        ms="b"
        model_size="base"
        ;;
    "small")
        ms="s" 
        model_size="small"
        ;;
    "large")
        ms="l"
        model_size="large"
        ;;
    *)
        echo "错误: 无效的模型大小 '$MODEL_SIZE'"
        echo "请选择: base, small, large"
        exit 1
        ;;
esac

echo "模型大小: $MODEL_SIZE (${ms})"
echo "潜在令牌数量: $NUM_LATENT_TOKENS"
echo "数据集: $DATASET_NAME"
echo "随机种子: $SPLIT_SEED"
echo "数据根目录: $DATA_ROOT"
echo "输出目录: $OUTPUT_ROOT"
echo "数据加载 worker 数: $NUM_WORKERS"
echo "batch size: $BATCH_SIZE"


# 启动微调训练
python modules/harmonizer/stage2_finetune/main_finetune_rep.py \
    --batch_size ${BATCH_SIZE} \
    --model vit_base_patch16 \
    --output_dir ${OUTPUT_ROOT}/stage2_finetune/harmonizer_vit${ms}_${NUM_LATENT_TOKENS} \
    --log_dir ${OUTPUT_ROOT}/stage2_finetune/harmonizer_vit${ms}_${NUM_LATENT_TOKENS} \
    --epochs 50 \
    --lr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 \
    --dist_eval \
    --nb_classes 2 \
    --dataset_name ${DATASET_NAME} \
    --split_seed ${SPLIT_SEED} \
    --data_path ${DATA_ROOT} \
    --num_workers ${NUM_WORKERS} \
    --pin_mem \
    --finetune checkpoints/harmonizer/model.pth
