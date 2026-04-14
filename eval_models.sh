#!/bin/bash

set -o pipefail

# LM Evaluation Harness 평가 스크립트
# - 실행 인자 파싱 없이, 모델별로 고정 명령을 순차 실행합니다.

export CUDA_VISIBLE_DEVICES=1  

# 평가할 모델 목록 (HuggingFace model ID)
model_list=(
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.1-8B-instruct"
)

export VLLM_USE_STANDALONE_COMPILE=0
# 시드 고정 (python, numpy, torch 모두 동일한 시드 사용)
# export CUBLAS_WORKSPACE_CONFIG=":4096:8"  # cuBLAS 결정론적 모드
# export PYTHONHASHSEED="42"

# 재현성 고정 시드 (python,numpy,torch,fewshot 모두 동일)
EVAL_SEED="42"

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/eval_${TIMESTAMP}.log"

echo "평가 시작 시간: $(date)" | tee "$LOG_FILE"
echo "고정 시드: $EVAL_SEED" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

total_models=${#model_list[@]}
current=0
total_tasks=3
TASK_INCLUDE_PATH="/home/yonsei_jong/lm-evaluation-harness/lm_eval/tasks"

for model in "${model_list[@]}"; do
    ((current++))
    # 동일한 결과 나오게 하는 시드.
    # MODEL_ARGS="pretrained=$model,seed=$EVAL_SEED,enforce_eager=True,enable_prefix_caching=False,max_num_seqs=1"
    MODEL_ARGS="pretrained=$model,seed=$EVAL_SEED,enforce_eager=True,enable_prefix_caching=False"

    # 전략 1) Instruct/Chat 모델: chat template + 5-shot
    # 전략 2) 로컬 SFT/베이스 모델: plain prompt (arc_challenge: 0-shot, gsm8k: 5-shot)
    Instruct_FLAGS=()
    if [[ "$model" == *"Instruct"* || "$model" == *"instruct"* || "$model" == *"-chat"* || "$model" == *"_chat"* ]]; then
        Instruct_FLAGS+=(--apply_chat_template)
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "[$current/$total_models] 모델 평가 중: $model" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    echo "[1/$total_tasks] task 평가 중: gsm8k" | tee -a "$LOG_FILE"
    if lm_eval --model vllm \
        --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.7 \
        --tasks gsm8k \
        --device cuda \
        --seed "$EVAL_SEED" \
        --batch_size 32 \
        --output_path eval_results \
        --log_samples \
        --num_fewshot 5 \
        "${Instruct_FLAGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ 완료: $model (task: gsm8k)" | tee -a "$LOG_FILE"
    else
        echo "✗ 실패: $model (task: gsm8k) - 종료 코드: $?" | tee -a "$LOG_FILE"
        exit 1
    fi


    echo "[2/$total_tasks] task 평가 중: arc-c" | tee -a "$LOG_FILE"
    if lm_eval --model vllm \
        --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.7 \
        --tasks arc_challenge \
        --seed "$EVAL_SEED" \
        --device cuda \
        --num_fewshot 5 \
        --batch_size 32 \
        --output_path eval_results \
        --log_samples \
        "${Instruct_FLAGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ 완료: $model (task: arc-c)" | tee -a "$LOG_FILE"
    else
        echo "✗ 실패: $model (task: arc-c) - 종료 코드: $?" | tee -a "$LOG_FILE"
        exit 1
    fi

   echo "[3/$total_tasks] task 평가 중: mmlu" | tee -a "$LOG_FILE"
    if lm_eval --model vllm \
        --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.7 \
        --tasks mmlu \
        --device cuda \
        --seed "$EVAL_SEED" \
        --batch_size 32 \
        --num_fewshot 5 \
        --output_path eval_results \
        --log_samples \
        "${Instruct_FLAGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ 완료: $model (task: mmlu)" | tee -a "$LOG_FILE"
    else
        echo "✗ 실패: $model (task: mmlu) - 종료 코드: $?" | tee -a "$LOG_FILE"
        exit 1
    fi

    # echo "[1/$total_tasks] task 평가 중: hendrycks_math_safe" | tee -a "$LOG_FILE"
    # if lm_eval --model vllm \
    #     --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.7 \
    #     --tasks "hendrycks_math_safe" \
    #     --include_path "$TASK_INCLUDE_PATH" \
    #     --seed "$EVAL_SEED" \
    #     --device cuda \
    #     --batch_size 32 \
    #     --output_path eval_results \
    #     --apply_chat_template \
    #     --num_fewshot 5 \
    #     --log_samples 2>&1 | tee -a "$LOG_FILE"; then
    #     echo "✓ 완료: $model (task: hendrycks_math_safe)" | tee -a "$LOG_FILE"
    # else
    #     echo "✗ 실패: $model (task: hendrycks_math_safe) - 종료 코드: $?" | tee -a "$LOG_FILE"
    #     exit 1
    # fi

done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "모든 평가 완료!" | tee -a "$LOG_FILE"
echo "총 실행: $((total_models * total_tasks))개 (모델 $total_models개 × task $total_tasks개)" | tee -a "$LOG_FILE"
echo "결과 경로: eval_results/" | tee -a "$LOG_FILE"
echo "로그 파일: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "평가 종료 시간: $(date)" | tee -a "$LOG_FILE"
