#!/bin/bash

set -o pipefail

# LM Evaluation Harness 평가 스크립트
# - 실행 인자 파싱 없이, 모델별로 고정 명령을 순차 실행합니다.

export CUDA_VISIBLE_DEVICES=0

# /tmp 이 noexec 마운트되어 있어 Triton .so 실행 불가 → 실행 가능한 경로로 우회
export TRITON_CACHE_DIR=/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/minseong/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

# 평가할 모델 목록 (HuggingFace model ID)
model_list=(
    "kmseong/llama2_7b_chat_mmlu_freeze_sn_lr5e-5"
    "kmseong/llama2_7b_chat_mmlu_freeze_rsn_lr5e-5"
    "kmseong/llama2-7b-chat-WaRP-mmlu-lr5e-5"
)

export VLLM_USE_STANDALONE_COMPILE=0
export VLLM_USE_DEEP_GEMM=0
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
total_tasks=2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_INCLUDE_PATH="$SCRIPT_DIR/lm_eval/tasks"

for model in "${model_list[@]}"; do
    ((current++))
    # 동일한 결과 나오게 하는 시드.
    # MODEL_ARGS="pretrained=$model,seed=$EVAL_SEED,enforce_eager=True,enable_prefix_caching=False,max_num_seqs=1"
    MODEL_ARGS="pretrained=$model,seed=$EVAL_SEED,enforce_eager=True,enable_prefix_caching=False"

    # 전략 1) Instruct/Chat 모델: chat template + 5-shot
    # 전략 2) 로컬 SFT/베이스 모델: plain prompt (arc_challenge: 0-shot, gsm8k: 5-shot)
    # Llama-2 계열: fewshot_as_multiturn=True와 호환 안 되므로 --apply_chat_template 제외
    Instruct_FLAGS=()
    # if [[ "$model" == *"Llama-2"* || "$model" == *"llama-2"* || "$model" == *"llama2"* ]]; then
    #     : # Llama-2는 --apply_chat_template 미사용 (fewshot_as_multiturn 비호환)
    #     echo "  - Llama-2 모델로 감지됨: --apply_chat_template 미사용"
    # el
    if [[ "$model" == *"Instruct"* || "$model" == *"instruct"* || "$model" == *"-chat"* || "$model" == *"_chat"* ]]; then
        Instruct_FLAGS+=(--apply_chat_template)
        echo "  - Instruct/Chat 모델로 감지됨: Instruct_FLAGS에 --apply_chat_template 추가"
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "[$current/$total_models] 모델 평가 중: $model" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    # echo "[1/$total_tasks] task 평가 중: gsm8k" | tee -a "$LOG_FILE"
    # if lm_eval --model vllm \
    #     --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.3 \
    #     --tasks gsm8k \
    #     --device cuda \
    #     --seed "$EVAL_SEED" \
    #     --batch_size 32 \
    #     --output_path eval_results \
    #     --log_samples \
    #     --num_fewshot 5 \
    #     "${Instruct_FLAGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    #     echo "✓ 완료: $model (task: gsm8k)" | tee -a "$LOG_FILE"
    # else
    #     echo "✗ 실패: $model (task: gsm8k) - 종료 코드: $?" | tee -a "$LOG_FILE"
    #     exit 1
    # fi


    # echo "[2/$total_tasks] task 평가 중: arc-c" | tee -a "$LOG_FILE"
    # if lm_eval --model vllm \
    #     --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.7 \
    #     --tasks arc_challenge_chat \
    #     --seed "$EVAL_SEED" \
    #     --device cuda \
    #     --num_fewshot 5 \
    #     --batch_size 32 \
    #     --output_path eval_results \
    #     --log_samples \
    #     --apply_chat_template 2>&1 | tee -a "$LOG_FILE"; then
    #     echo "✓ 완료: $model (task: arc-c)" | tee -a "$LOG_FILE"
    # else
    #     echo "✗ 실패: $model (task: arc-c) - 종료 코드: $?" | tee -a "$LOG_FILE"
    #     exit 1
    # fi

    # echo "[1/$total_tasks] task 평가 중: medqa_4options" | tee -a "$LOG_FILE"
    # if lm_eval --model vllm \
    #     --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.4 \
    #     --tasks medqa_4options \
    #     --include_path "$TASK_INCLUDE_PATH" \
    #     --device cuda \
    #     --seed "$EVAL_SEED" \
    #     --batch_size 32 \
    #     --num_fewshot 5 \
    #     --output_path eval_results \
    #     --log_samples \
    #     "${Instruct_FLAGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    #     echo "✓ 완료: $model (task: medqa_4options)" | tee -a "$LOG_FILE"
    # else
    #     echo "✗ 실패: $model (task: medqa_4options) - 종료 코드: $?" | tee -a "$LOG_FILE"
    #     exit 1
    # fi

    echo "[2/$total_tasks] task 평가 중: mmlu" | tee -a "$LOG_FILE"
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
    #     --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.4 \
    #     --tasks "hendrycks_math_safe" \
    #     --include_path "$TASK_INCLUDE_PATH" \
    #     --seed "$EVAL_SEED" \
    #     --device cuda \
    #     --batch_size 32 \
    #     --output_path eval_results \
    #     --num_fewshot 5 \
    #     "${Instruct_FLAGS[@]}" \
    #     --log_samples 2>&1 | tee -a "$LOG_FILE"; then
    #     echo "✓ 완료: $model (task: hendrycks_math_safe)" | tee -a "$LOG_FILE"
    # else
    #     echo "✗ 실패: $model (task: hendrycks_math_safe) - 종료 코드: $?" | tee -a "$LOG_FILE"
    #     exit 1
    # fi

    # mbpp 평가 전략:
    #   - mbpp + --apply_chat_template: 학습 포맷([INST]...[/INST])과 평가 포맷 일치
    #   - Llama-2 계열도 MBPP 평가에서는 apply_chat_template 사용
    #     (다른 task에서는 fewshot_as_multiturn 비호환으로 제외했지만,
    #      mbpp는 1-shot + [DONE] stop token 방식으로 정상 동작)
    #   - chat 모델 여부: 모델명에 chat/instruct/llama2/_chat 포함 시 적용
    #   - base 모델: apply_chat_template 없음
    # 실제 코드 실행이 필요하므로 HF_ALLOW_CODE_EVAL=1 + --confirm_run_unsafe_code 필수
    # 컨텍스트 초과 방지: --num_fewshot 0 (0-shot)
    #   - 1-shot 예시가 ~3800토큰 차지 → context+gen(512)=4492 > 4096 → 좌측 truncation 발생
    #   - MBPP-FT 모델은 학습 데이터를 이미 봤으므로 0-shot에서도 포맷 파악 가능

    # MBPP_TASK="mbpp"
    # MBPP_GEN_KWARGS="max_gen_toks=512"
    # if [[ "$model" == *"chat"* || "$model" == *"Chat"* || \
    #       "$model" == *"instruct"* || "$model" == *"Instruct"* || \
    #       "$model" == *"llama2"* || "$model" == *"llama-2"* || "$model" == *"Llama-2"* ]]; then
    #     MBPP_FLAGS=(--apply_chat_template)
    #     echo "  - Chat/Instruct 모델: MBPP 평가에 --apply_chat_template 적용 (학습 포맷 일치)"
    # else
    #     MBPP_FLAGS=()
    # fi
    # echo "[2/$total_tasks] task 평가 중: $MBPP_TASK" | tee -a "$LOG_FILE"
    # if HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm \
    #     --model_args "$MODEL_ARGS",tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.4 \
    #     --tasks "$MBPP_TASK" \
    #     --seed "$EVAL_SEED" \
    #     --device cuda \
    #     --batch_size 32 \
    #     --output_path eval_results \
    #     --num_fewshot 0 \
    #     --gen_kwargs "$MBPP_GEN_KWARGS" \
    #     --confirm_run_unsafe_code \
    #     "${MBPP_FLAGS[@]}" \
    #     --log_samples 2>&1 | tee -a "$LOG_FILE"; then
    #     echo "✓ 완료: $model (task: $MBPP_TASK)" | tee -a "$LOG_FILE"
    # else
    #     echo "✗ 실패: $model (task: $MBPP_TASK) - 종료 코드: $?" | tee -a "$LOG_FILE"
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

# === 결과 요약 파일 생성 ===
RESULTS_FILE="logs/eval_${TIMESTAMP}_results.log"

python3 << PYEOF
import re

log_file = "$LOG_FILE"
results_file = "$RESULTS_FILE"

with open(log_file, 'r') as f:
    lines = f.readlines()

n = len(lines)
current_model = None
blocks = []  # list of (model_header, table_lines, completion_line)

i = 0
while i < n:
    line = lines[i].rstrip('\n')

    # 모델 헤더 라인 감지
    if '모델 평가 중:' in line:
        current_model = line

    # 완료/실패 라인 감지
    if line.startswith('✓') or line.startswith('✗'):
        # 바로 위의 마지막 테이블 블록을 역방향으로 탐색
        j = i - 1
        while j >= 0 and lines[j].strip() == '':
            j -= 1
        table_end = j
        while j >= 0 and lines[j].startswith('|'):
            j -= 1
        table_start = j + 1

        if table_start <= table_end and lines[table_start].startswith('|'):
            table_lines = [lines[k].rstrip('\n') for k in range(table_start, table_end + 1)]
        else:
            table_lines = []

        blocks.append((current_model, table_lines, line))

    i += 1

with open(results_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("평가 결과 요약\n")
    f.write("=" * 60 + "\n\n")

    prev_model = None
    for model_header, table_lines, completion_line in blocks:
        if model_header != prev_model:
            if prev_model is not None:
                f.write("\n")
            if model_header:
                f.write(model_header + "\n")
                f.write("-" * 40 + "\n")
            prev_model = model_header

        for tl in table_lines:
            f.write(tl + "\n")
        if table_lines:
            f.write("\n")

        f.write(completion_line + "\n")

    f.write("\n" + "=" * 60 + "\n")

print("결과 요약 파일 저장 완료: " + results_file)
PYEOF

echo "결과 요약: $RESULTS_FILE" | tee -a "$LOG_FILE"
