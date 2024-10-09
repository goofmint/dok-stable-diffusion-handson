#!/bin/bash

set -ue
shopt -s nullglob

export TZ=${TZ:-Asia/Tokyo}

if [ -z "${SAKURA_ARTIFACT_DIR:-}" ]; then
  echo "Environment variable SAKURA_ARTIFACT_DIR is not set" >&2
  exit 1
fi

if [ -z "${SAKURA_TASK_ID:-}" ]; then
  echo "Environment variable SAKURA_TASK_ID is not set" >&2
  exit 1
fi

if [ -z "${PROMPT:-}" ]; then
  echo "Environment variable PROMPT is not set" >&2
  exit 1
fi

pushd /stable-diffusion
  if [ ! -e fuduki_mix ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://huggingface.co/Kotajiro/fuduki_mix
    pushd fuduki_mix
      git lfs pull -I /fuduki_mix_v20.safetensors
    popd
  fi
  
  python3 runner.py \
    --batch="${BATCH:-1}" \
    --height="${HEIGHT:-1024}" \
    --model="./fuduki_mix/fuduki_mix_v20.safetensors" \
    --variant="single" \
    --negative="${NEGATIVE_PROMPT:-}" \
    --num="${NUM_IMAGES:-1}" \
    --output="${SAKURA_ARTIFACT_DIR}" \
    --prefix="${SAKURA_TASK_ID}" \
    --prompt="${PROMPT}" \
    --s3-bucket="${S3_BUCKET:-}" \
    --s3-endpoint="${S3_ENDPOINT:-}" \
    --s3-secret="${S3_SECRET:-}" \
    --s3-token="${S3_TOKEN:-}" \
    --seed="${SEED:--1}" \
    --steps="${STEPS:-20}" \
    --width="${WIDTH:-1024}"
  
popd