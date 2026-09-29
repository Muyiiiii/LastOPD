#!/bin/bash
# Official 8-benchmark evaluation of a trained checkpoint.
#   bash scripts/lastopd/eval.sh <fsdp actor checkpoint dir | HF model dir> <run name> [gpu ids, e.g. 0,1,2,3]
# Steps: (1) merge FSDP shards to HF (skipped if <dir> already holds *.safetensors)  (2) vLLM generation
#        (3) rule grading  (4) paper-table avg@k / best@k  ->  $OUT_DIR/official_eval/<run name>/table_scores.json
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; E=$ROOT/scripts/lastopd/eval
SRC=${1:?checkpoint or model dir}; NAME=${2:?run name}; GPUS=${3:-0,1,2,3,4,5,6,7}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
if ls "$SRC"/*.safetensors >/dev/null 2>&1; then MODEL=$SRC; else
  MODEL=$OUT_DIR/merged/$NAME
  [ -n "$(ls $MODEL/*.safetensors 2>/dev/null)" ] || (cd "$ROOT/verl" && python3 -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$MODEL")
fi
MODEL_PATH=$MODEL OUT_NAME=$NAME GPU_IDS=$GPUS MATH500_N=8 AIMO_N=16 MINERVA_N=4 OLYMPIAD_N=4 AMC23_N=16 GSM8K_N=4 python3 $E/gen_vllm_official.py
MODEL_PATH=$MODEL OUT_NAME=$NAME python3 $E/grade_official.py
OUT_NAME=$NAME python3 $E/table_scores.py
