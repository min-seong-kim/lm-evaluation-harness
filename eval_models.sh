#!/bin/bash
set -o pipefail

# LM Evaluation Harness 평가 스크립트 - 여러 모델과 task 자동 실행

# ========================================
# 설정: 평가할 모델과 task 목록
# ========================================

# 평가할 모델 목록 (HuggingFace model ID)
model_list=(
    "meta-llama/Llama-3.2-3B-Instruct"
    # "/lustre/gokms0509/Safety-WaRP-LLM/checkpoints/phase0_20260323_114637"
    # "/root/lm-evaluation-harness/gsm8k_llama32_3b_full_sft"
    # "/root/lm-evaluation-harness/gsm8k_llama32_3b_full_sft_20260325_062640"
    "/root/lm-evaluation-harness/gsm8k_llama32_3b_full_sft_flexible_20260325_113443"
    # "meta-llama/Llama-3.2-3B"
    # "kmseong/Llama3.2-3B-gsm8k-full-FT"
    # "kmseong/Llama-3.2-3B-SSFT"
    # "kmseong/Llama-3.2-3B-SafeLoRA"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "kmseong/Llama-3.2-3B-gsm8k_ft_after-rsn-tuned-freeze_rsn_10"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-all-layer-0.05"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-all-layer-0.15"
    # "kmseong/Llama-3.2-3B-only-sn-tuned"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-wikipedia-base-start-perlayer"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-rsn-tuned-start"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-per-layer"
    # "kmseong/Llama-3.2-3B-instruct-metamath-finetuned"
    # "kmseong/safety-warp-Llama-3.2-3b-instruct-phase3-perlayer-metamath"
    # "kmseong/Llama-3.2-3B-instruct-SafeLoRA-1"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-non-freeze"
)

# 평가할 task 목록
tasks_list=(
    # "arc_challenge"
    # "mmlu"
    "gsm8k"
    # "gsm8k-cot"
    # "mathqa"
)

# 평가 설정
DEVICE="cuda"
BATCH_SIZE=auto
OUTPUT_PATH="eval_results"
APPLY_CHAT_TEMPLATE="true"
FEWSHOT_AS_MULTITURN="true"
NUM_FEWSHOT="5"
LIMIT=""
LOG_SAMPLES="false"
GEN_KWARGS="temperature=0,top_p=1,max_gen_toks=512"
VLLM_USE_STANDALONE_COMPILE="0"

# 실행 인자 파싱
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply_chat_template)
            APPLY_CHAT_TEMPLATE="$2"
            shift 2
            ;;
        --fewshot_as_multiturn)
            FEWSHOT_AS_MULTITURN="$2"
            shift 2
            ;;
        --num_fewshot)
            NUM_FEWSHOT="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --log_samples)
            LOG_SAMPLES="$2"
            shift 2
            ;;
        --gen_kwargs)
            GEN_KWARGS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --vllm_use_standalone_compile)
            VLLM_USE_STANDALONE_COMPILE="$2"
            shift 2
            ;;
    esac
done

# ========================================
# 평가 실행
# ========================================

# 로그 디렉토리 생성
mkdir -p logs

# 로그 파일 이름: task명_현재시각.log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/eval_${TIMESTAMP}.log"

# 스크립트 시작 시간 기록
echo "평가 시작 시간: $(date)" | tee "$LOG_FILE"
echo "결과 디렉토리: $OUTPUT_PATH" | tee -a "$LOG_FILE"
echo "apply_chat_template: $APPLY_CHAT_TEMPLATE" | tee -a "$LOG_FILE"
echo "fewshot_as_multiturn: $FEWSHOT_AS_MULTITURN" | tee -a "$LOG_FILE"
echo "num_fewshot: $NUM_FEWSHOT" | tee -a "$LOG_FILE"
echo "limit: ${LIMIT:-all}" | tee -a "$LOG_FILE"
echo "log_samples: $LOG_SAMPLES" | tee -a "$LOG_FILE"
echo "gen_kwargs: ${GEN_KWARGS:-default}" | tee -a "$LOG_FILE"
echo "vllm_use_standalone_compile: $VLLM_USE_STANDALONE_COMPILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

total_models=${#model_list[@]}
total_tasks=${#tasks_list[@]}
current=0

for task in "${tasks_list[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Task: $task" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    
    for model in "${model_list[@]}"; do
        ((current++))
        echo "" | tee -a "$LOG_FILE"
        echo "[$current/$((total_models * total_tasks))] 모델 평가 중: $model" | tee -a "$LOG_FILE"
        echo "Task: $task" | tee -a "$LOG_FILE"
        echo "명령 실행 중..." | tee -a "$LOG_FILE"

        MODEL_ARGS="pretrained=$model"
        EXTRA_ARGS=()
        OPTIONAL_ARGS=()
        if [[ "$APPLY_CHAT_TEMPLATE" == "true" ]]; then
            EXTRA_ARGS+=(--apply_chat_template)
        fi
        if [[ "$FEWSHOT_AS_MULTITURN" == "true" ]]; then
            EXTRA_ARGS+=(--fewshot_as_multiturn "$FEWSHOT_AS_MULTITURN")
        fi
        if [[ -n "$LIMIT" ]]; then
            OPTIONAL_ARGS+=(--limit "$LIMIT")
        fi
        if [[ "$LOG_SAMPLES" == "true" ]]; then
            OPTIONAL_ARGS+=(--log_samples)
        fi
        if [[ -n "$GEN_KWARGS" ]]; then
            OPTIONAL_ARGS+=(--gen_kwargs "$GEN_KWARGS")
        fi

        export VLLM_USE_STANDALONE_COMPILE="$VLLM_USE_STANDALONE_COMPILE"
        
        if lm_eval --model vllm \
            --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
            --tasks "$task" \
            --device "$DEVICE" \
            --batch_size "$BATCH_SIZE" \
            --num_fewshot "$NUM_FEWSHOT" \
            "${EXTRA_ARGS[@]}" \
            "${OPTIONAL_ARGS[@]}" \
            --output_path "$OUTPUT_PATH" 2>&1 | tee -a "$LOG_FILE"; then
            echo "✓ 완료: $model (task: $task)" | tee -a "$LOG_FILE"
        else
            echo "✗ 실패: $model (task: $task) - 종료 코드: $?" | tee -a "$LOG_FILE"
            exit 1
        fi
    done
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "모든 평가 완료!" | tee -a "$LOG_FILE"
echo "총 실행: $((total_models * total_tasks))개 (모델 $total_models개 × task $total_tasks개)" | tee -a "$LOG_FILE"
echo "결과 경로: $OUTPUT_PATH/" | tee -a "$LOG_FILE"
echo "로그 파일: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "평가 종료 시간: $(date)" | tee -a "$LOG_FILE"
