#!/bin/bash

# LM Evaluation Harness 평가 스크립트 - 여러 모델과 task 자동 실행

# ========================================
# 설정: 평가할 모델과 task 목록
# ========================================

# 평가할 모델 목록 (HuggingFace model ID)
model_list=(
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "kmseong/Llama-3.2-3B-gsm8k_ft_after-rsn-tuned-freeze_rsn_10"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-all-layer-0.05"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-all-layer-0.15"
    # "kmseong/Llama-3.2-3B-only-sn-tuned"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-wikipedia-base-start-perlayer"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-rsn-tuned-start"
    # "kmseong/safety-warp-Llama-3.2-3b-phase3-per-layer"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "kmseong/Llama-3.2-3B-instruct-metamath-finetuned"
    # "kmseong/safety-warp-Llama-3.2-3b-instruct-phase3-perlayer-metamath"
    "kmseong/Llama-3.2-3B-instruct-SafeLoRA-1"
)

# 평가할 task 목록
tasks_list=(
    "gsm8k"
    "mmlu"
)

# 평가 설정
DEVICE="cuda"
BATCH_SIZE=8
OUTPUT_PATH="eval_results"

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
        
        if lm_eval --model hf \
            --model_args pretrained="$model" \
            --tasks "$task" \
            --device "$DEVICE" \
            --batch_size "$BATCH_SIZE" \
            --log_samples \
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
