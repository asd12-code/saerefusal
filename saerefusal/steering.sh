#!/usr/bin/env bash

set -uo pipefail

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120
export PYTHONUNBUFFERED=1

MODEL_PATH="google/gemma-3-27b-it"
DEVICE="cuda:5"
BATCH_SIZE=64

MODEL_TAG="${MODEL_PATH//\//-}"

mkdir -p logs

# AlpacaEval unsafe
# python -u steering.py \
#   --model_path "$MODEL_PATH" \
#   --device "$DEVICE" \
#   --batch_size "$BATCH_SIZE" \
#   --steering_dataset alpaca_eval \
#   --method ranked \
#   --ranked_feature_topk 5 \
#   > "logs/${MODEL_TAG}_alpaca_eval_unsafe_ranked.log" 2>&1

# AlpacaEval safe
# python -u steering.py \
#   --model_path "$MODEL_PATH" \
#   --device "$DEVICE" \
#   --batch_size "$BATCH_SIZE" \
#   --steering_dataset alpaca_eval \
#   --use_safe \
#   --method ranked \
#   --ranked_feature_topk 5 \
#   > "logs/${MODEL_TAG}_alpaca_eval_safe_ranked.log" 2>&1

# WildGuardMix unsafe
# python -u steering.py \
#   --model_path "$MODEL_PATH" \
#   --device "$DEVICE" \
#   --batch_size "$BATCH_SIZE" \
#   --steering_dataset wildguardmix \
#   --method ranked \
#   --ranked_feature_topk 5 \
#   > "logs/${MODEL_TAG}_wildguardmix_unsafe_ranked.log" 2>&1

# WildGuardMix safe
# python -u steering.py \
#   --model_path "$MODEL_PATH" \
#   --device "$DEVICE" \
#   --batch_size "$BATCH_SIZE" \
#   --steering_dataset wildguardmix \
#   --use_safe \
#   --method ranked \
#   --ranked_feature_topk 5 \
#   > "logs/${MODEL_TAG}_wildguardmix_safe_ranked.log" 2>&1

# XSTest unsafe
# python -u steering.py \
#   --model_path "$MODEL_PATH" \
#   --device "$DEVICE" \
#   --batch_size "$BATCH_SIZE" \
#   --steering_dataset xstest \
#   --method ranked \
#   --ranked_feature_topk 5 \
#   > "logs/${MODEL_TAG}_xstest_unsafe_ranked.log" 2>&1

# XSTest safe
python -u steering.py \
  --model_path "$MODEL_PATH" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --steering_dataset xstest \
  --use_safe \
  --method ranked \
  --ranked_feature_topk 5 \
  > "logs/${MODEL_TAG}_xstest_safe_ranked.log" 2>&1