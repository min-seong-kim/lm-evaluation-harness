#!/bin/bash
set -o pipefail

# LM Evaluation Harness 평가 스크립트
# - 실행 인자 파싱 없이, 모델별로 고정 명령을 순차 실행합니다.

# 평가할 모델 목록 (HuggingFace model ID)
model_list=(
    # "kmseong/llama3.2-3b-WaRP-utility-basis-safety-FT-no-rotation"
    # "meta-llama/Llama-3.2-3B"
    # "kmseong/Llama-3.2-3B-SSFT"
    # "kmseong/llama3.2-3b-sn-tune-1.3p"
    # "kmseong/llama3.2-3b-WaRP-utility-basis-safety-FT"
    # "kmseong/Llama3.2-3B-gsm8k-full-FT"
    "kmseong/Llama3.2-3B-gsm8k-fullft-atfter-ssft"
    "kmseong/Llama-3.2-3B-gsm8k-ft-after-rsn-tuned-freeze-sn"

    # "kmseong/Llama-3.2-3B-only-sn-tuned"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-wikipedia-base-start-perlayer"
    # "kmseong/Llama-3.2-3B-only-rsn-tuned_10"
    # "kmseong/Llama-3.2-3B-gsm8k_ft_after-rsn-tuned-freeze_rsn_10"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-whole-layer"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-whole-layer-non-freeze"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-whole-layer-all_layer"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-per-layer"
    "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-non-freeze"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-all-layer"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-rsn-tuned-start"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-no_rotation-per-layer"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-no_rotation-per-layer2"
    # "/root/lm-evaluation-harness/MATH_llama32_3b_full_sft_20260326_123531"
    # "kmseong/Llama-3.2-3B-instruct-SafeLoRA2"
    # "kmseong/safety-warp-Llama-3.2-3b-instruct-warp-MATH2"
    # "kmseong/Llama-3.2-3B-only-rsn-tuned"
    # "kmseong/llama3.2-3b-WaRP-utility-basis-safety-FT-freeze"
)

export VLLM_USE_STANDALONE_COMPILE="0"
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
TASK_NAME="hendrycks_math_safe"
TASK_INCLUDE_PATH="/root/lm-evaluation-harness/lm_eval/tasks"

for model in "${model_list[@]}"; do
    ((current++))
    # 동일한 결과 나오게 하는 시드.
    # MODEL_ARGS="pretrained=$model,seed=$EVAL_SEED,enforce_eager=True,enable_prefix_caching=False,max_num_seqs=1"
    # MODEL_ARGS="pretrained=$model,seed=$EVAL_SEED,enforce_eager=True,enable_prefix_caching=False"
    MODEL_ARGS="pretrained=$model,seed=$EVAL_SEED"
    EXTRA_FLAGS=()
    STRATEGY_LABEL=""

    # 전략 1) Instruct/Chat 모델: chat template + 5-shot
    # 전략 2) 로컬 SFT/베이스 모델: plain prompt (arc_challenge: 0-shot, gsm8k: 5-shot)
    GSM8K_FLAGS=()
    if [[ "$model" == *"Instruct"* || "$model" == *"-chat"* || "$model" == *"_chat"* ]]; then
        EXTRA_FLAGS+=(--apply_chat_template --num_fewshot 5)
        GSM8K_FLAGS+=(--apply_chat_template --fewshot_as_multiturn --num_fewshot 5)
        STRATEGY_LABEL="chat-template, 5-shot"
    else
        EXTRA_FLAGS+=(--num_fewshot 0)
        GSM8K_FLAGS+=(--num_fewshot 5)
        STRATEGY_LABEL="plain-prompt, 0-shot (arc) / 5-shot (gsm8k)"
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "[$current/$total_models] 모델 평가 중: $model" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    echo "[1/$total_tasks] task 평가 중: gsm8k" | tee -a "$LOG_FILE"
    echo "명령어: lm_eval --model vllm --model_args \"$MODEL_ARGS,tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8\" --tasks gsm8k --device cuda --batch_size 8 --output_path eval_results --log_samples$CHAT_FLAGS_STR" | tee -a "$LOG_FILE"
    if lm_eval --model vllm \
        --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
        --tasks gsm8k \
        --device cuda \
        --seed "$EVAL_SEED" \
        --batch_size 8 \
        --output_path eval_results \
        --log_samples \
        "${GSM8K_FLAGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ 완료: $model (task: gsm8k)" | tee -a "$LOG_FILE"
    else
        echo "✗ 실패: $model (task: gsm8k) - 종료 코드: $?" | tee -a "$LOG_FILE"
        exit 1
    fi

    # echo "[2/$total_tasks] task 평가 중: mmlu" | tee -a "$LOG_FILE"
    # echo "명령어: lm_eval --model vllm --model_args \"$MODEL_ARGS,tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8\" --tasks mmlu --device cuda --batch_size 8 --output_path eval_results --log_samples" | tee -a "$LOG_FILE"
    # if lm_eval --model vllm \
    #     --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
    #     --tasks mmlu \
    #     --device cuda \
    #     --seed "$EVAL_SEED" \
    #     --batch_size 8 \
    #     --output_path eval_results \
    #     --log_samples 2>&1 | tee -a "$LOG_FILE"; then
    #     echo "✓ 완료: $model (task: mmlu)" | tee -a "$LOG_FILE"
    # else
    #     echo "✗ 실패: $model (task: mmlu) - 종료 코드: $?" | tee -a "$LOG_FILE"
    #     exit 1
    # fi

    # echo "[3/$total_tasks] task 평가 중: arc-c" | tee -a "$LOG_FILE"
    # echo "명령어: lm_eval --model vllm --model_args \"$MODEL_ARGS,tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8\" --tasks arc-c --device cuda --batch_size 8 --output_path eval_results --log_samples" | tee -a "$LOG_FILE"
    # if lm_eval --model vllm \
    #     --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
    #     --tasks arc_challenge \
    #     --seed "$EVAL_SEED" \
    #     --device cuda \
    #     --batch_size 8 \
    #     --output_path eval_results \
    #     --log_samples  2>&1 | tee -a "$LOG_FILE"; then
    #     echo "✓ 완료: $model (task: arc-c)" | tee -a "$LOG_FILE"
    # else
    #     echo "✗ 실패: $model (task: arc-c) - 종료 코드: $?" | tee -a "$LOG_FILE"
    #     exit 1
    # fi

    # echo "[1/$total_tasks] task 평가 중: $TASK_NAME" | tee -a "$LOG_FILE"
    # echo "평가 전략: $STRATEGY_LABEL" | tee -a "$LOG_FILE"
    # echo "명령어: lm_eval --model vllm --model_args \"$MODEL_ARGS,tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8\" --tasks $TASK_NAME --include_path $TASK_INCLUDE_PATH --seed $EVAL_SEED --device cuda --batch_size 8 --output_path eval_results --log_samples ${EXTRA_FLAGS[*]}" | tee -a "$LOG_FILE"
    # if lm_eval --model vllm \
    #     --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
    #     --tasks "$TASK_NAME" \
    #     --include_path "$TASK_INCLUDE_PATH" \
    #     --seed "$EVAL_SEED" \
    #     --device cuda \
    #     --batch_size 8 \
    #     --output_path eval_results \
    #     --log_samples 2>&1 | tee -a "$LOG_FILE"; then
    #     echo "✓ 완료: $model (task: $TASK_NAME)" | tee -a "$LOG_FILE"
    # else
    #     echo "✗ 실패: $model (task: $TASK_NAME) - 종료 코드: $?" | tee -a "$LOG_FILE"
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
