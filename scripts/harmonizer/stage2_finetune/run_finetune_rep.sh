#!/bin/bash

# 检查参数
if [ $# -ne 6 ] && [ $# -ne 4 ]; then
    echo "用法: $0 <模型大小> <num_latent_tokens> <数据集> <随机种子> [数据根目录 输出目录]"
    echo "模型大小: base, small, large"
    echo "数据集: 数据集名称"
    echo "随机种子: 用于数据分割的随机种子"
    echo "数据根目录: 例如 /p/project1/hai_1024/data (可用 DATA_ROOT 覆盖)"
    echo "输出目录: 例如 /p/project1/hai_1024/Brain-Harmony/experiments (可用 OUTPUT_ROOT 覆盖)"
    echo "例如: $0 base 128 ADNI 42 /p/project1/hai_1024/data /p/project1/hai_1024/Brain-Harmony/experiments"
    exit 1
fi

# 获取参数
MODEL_SIZE=$1
NUM_LATENT_TOKENS=$2
DATASET_NAME=$3
SPLIT_SEED=$4
if [ $# -eq 6 ]; then
    DATA_ROOT=$5
    OUTPUT_ROOT=$6
else
    DATA_ROOT="${DATA_ROOT:-}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-}"
fi
if [ -z "${DATA_ROOT}" ] || [ -z "${OUTPUT_ROOT}" ]; then
    echo "错误: DATA_ROOT 和 OUTPUT_ROOT 不能为空（可作为参数或环境变量提供）"
    exit 1
fi

# Default threading policy to avoid CPU oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-${SRUN_CPUS_PER_TASK:-1}}"
if [ -z "${NUM_WORKERS:-}" ]; then
    NUM_WORKERS=$(( SRUN_CPUS_PER_TASK / 4 ))
    if (( NUM_WORKERS < 1 )); then NUM_WORKERS=1; fi
fi
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-50}"
NB_CLASSES=2

if [ -n "${SLURM_PROCID:-}" ] && [ -z "${RANK:-}" ]; then
    export WORLD_SIZE="${SLURM_NTASKS:-1}"
    export RANK="${SLURM_PROCID}"
    export LOCAL_RANK="${SLURM_LOCALID:-0}"
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES#*,}" = "${CUDA_VISIBLE_DEVICES}" ]; then
    export LOCAL_RANK=0
fi


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
echo "数据加载 worker 数: $NUM_WORKERS (cpus-per-task=${SRUN_CPUS_PER_TASK:-unset})"
echo "batch size: $BATCH_SIZE"
echo "epochs: $EPOCHS"
if [ -n "${RANK:-}" ]; then
    echo "rank ${RANK} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "rank ${RANK} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
fi
case $DATASET_NAME in
    "SEEDV")
        NB_CLASSES=5
        ;;
    "ADNI"|"ADHD200"|"LEMON_fMRI")
        NB_CLASSES=2
        ;;
    *)
        echo "警告: 未知数据集 '$DATASET_NAME'，默认使用 NB_CLASSES=2"
        NB_CLASSES=2
        ;;
esac
echo "分类类别数: $NB_CLASSES"


# 启动微调训练
python modules/harmonizer/stage2_finetune/main_finetune_rep.py \
    --batch_size ${BATCH_SIZE} \
    --model vit_base_patch16 \
    --output_dir ${OUTPUT_ROOT}/stage2_finetune/harmonizer_vit${ms}_${NUM_LATENT_TOKENS} \
    --log_dir ${OUTPUT_ROOT}/stage2_finetune/harmonizer_vit${ms}_${NUM_LATENT_TOKENS} \
    --epochs ${EPOCHS} \
    --lr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 \
    --dist_eval \
    --nb_classes ${NB_CLASSES} \
    --dataset_name ${DATASET_NAME} \
    --split_seed ${SPLIT_SEED} \
    --data_path ${DATA_ROOT} \
    --num_workers ${NUM_WORKERS} \
    --pin_mem \
    --finetune checkpoints/harmonizer/model.pth
