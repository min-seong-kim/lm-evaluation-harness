#!/bin/bash

# gsm8k 평가 스크립트 - 여러 모델 순차 실행

set -e  # 에러 발생 시 스크립트 중단

echo "=========================================="
echo "GSM8K 평가 시작"
echo "=========================================="

# 베이스 모델
echo ""
echo "1. 베이스 모델 평가 중..."
lm_eval --model hf \
    --model_args pretrained=/root/lm-evaluation-harness/gsm8k_llama32_3b_full_sft \
    --tasks gsm8k \
    --device cuda \
    --batch_size 8 \
    --log_samples \
    --output_path eval_results

# 첫 번째 커스텀 모델
# echo ""
# echo "2. phase3-per-layer 모델 평가 중..."
# lm_eval --model hf \
#     --model_args pretrained=kmseong/Llama-3.2-3B-gsm8k-ft-after-rsn-tuned-origin-re \
#     --tasks gsm8k \
#     --device cuda \
#     --batch_size 8 \
#     --log_samples \
#     --output_path eval_results

# # 두 번째 커스텀 모델
# echo ""
# echo "3. phase3-non-freeze 모델 평가 중..."
# lm_eval --model hf \
#     --model_args pretrained=kmseong/Llama-3.2-3B-gsm8k-ft-after-rsn-tuned-freeze-sn \
#     --tasks gsm8k \
#     --device cuda \
#     --batch_size 8 \
#     --log_samples \
#     --output_path eval_results

# echo "4. phase3-non-freeze 모델 평가 중..."
# lm_eval --model hf \
#     --model_args pretrained=kmseong/Llama-3.2-3B-gsm8k-ft-after-rsn-tuned \
#     --tasks gsm8k \
#     --device cuda \
#     --batch_size 8 \
#     --log_samples \
#     --output_path eval_results

echo ""
echo "=========================================="
echo "모든 평가 완료!"
echo "결과 경로: results/"
echo "=========================================="
