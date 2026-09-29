#!/bin/bash
# =============================================================================
# LastOPD training launcher (single node, verl + vLLM).
#
# Distills a teacher into a student on the student's own rollouts with
#   L = token_coef(t) * L_token  +  latent_coef(t) * L_latent
#   L_token  : reverse-KL on the student's top-k tokens (OPD, TOP_K_STRATEGY=only_stu)
#   L_latent : normalized MSE between MLP(h_S^{last}) and h_T^{last} on ALL response tokens
#   crossfade: latent_coef 1->0 over REP_COEF_DECAY_STEPS, token_coef 0->1 over TOKEN_COEF_RAMP_STEPS
#
# Required env:  ACTOR_MODEL_PATH (student), REWARD_MODEL_PATH (teacher), DATA_DIR, OUT_DIR
# Optional env:  see the "knobs" block below.  Every knob has the paper default.
# Example:       see scripts/lastopd/cells.sh
# =============================================================================
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR=${DATA_DIR:?"set DATA_DIR (contains dapo-math-17k.parquet and test_data/<bench>/test.parquet)"}
OUT_DIR=${OUT_DIR:?"set OUT_DIR (checkpoints / logs / validation dumps go here)"}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:?"set ACTOR_MODEL_PATH (student)"}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:?"set REWARD_MODEL_PATH (teacher)"}
RUN_TAG=${RUN_TAG:-lastopd}
cd "$ROOT/verl"
mkdir -p "$OUT_DIR/logs/terminal"

# ---- method knobs (paper defaults = LastOPD) ----
export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-True}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1000}      # latent loss weight
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-last}   # last | all | even | odd
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-all}  # all | last_k | first_k | last
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1000}
export REP_PROJECTOR_MODE=${REP_PROJECTOR_MODE:-full}             # full (MLP/linear d_S->d_T) | low_rank (OPRD-Bridge)
export REP_FULL_PROJECTOR=${REP_FULL_PROJECTOR:-mlp}              # mlp | linear
export REP_COEF_DECAY_STEPS=${REP_COEF_DECAY_STEPS:-10}           # crossfade: latent 1->0 (0 = always on)
export TOKEN_COEF_RAMP_STEPS=${TOKEN_COEF_RAMP_STEPS:-10}         # crossfade: token 0->1 (0 = always on)
export REP_SCHED_STYLE=${REP_SCHED_STYLE:-linear}                 # linear | hard
export MODE=${MODE:-combined}                                     # combined (token+latent) | rep_only (latent only)
if [ "$MODE" = "combined" ]; then
  export REP_DISTILLATION_ONLY=False; export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
else
  export REP_DISTILLATION_ONLY=True;  export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
fi
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}                 # reverse token loss on student top-k
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
export REP_TEACHER_CACHE_DTYPE=${REP_TEACHER_CACHE_DTYPE:-fp32}   # bf16 halves the host-side teacher-hidden cache

# ---- chassis (paper) ----
export MODEL_DTYPE=${MODEL_DTYPE:-fp32}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
export N_RESPONSES=${N_RESPONSES:-4}
export VAL_N=${VAL_N:-8}
export TOTAL_STEPS=${TOTAL_STEPS:-62}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-62}
export N_GPUS=${N_GPUS:-8}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH ))
PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768 ))

# ---- data ----
TRAIN_DATASET=${TRAIN_DATASET:-$DATA_DIR/dapo-math-17k.parquet}
TEST_DATA_DIR=${TEST_DATA_DIR:-$DATA_DIR/test_data}
if [ "${VAL_MATH500_ONLY:-1}" = "1" ]; then
  TEST_DATASET='["'"$TEST_DATA_DIR"'/MATH-500/test.parquet"]'
else
  TEST_DATASET='["'"$TEST_DATA_DIR"'/AMC23/test.parquet", "'"$TEST_DATA_DIR"'/AIME24/test.parquet", "'"$TEST_DATA_DIR"'/AIME25/test.parquet", "'"$TEST_DATA_DIR"'/MATH-500/test.parquet"]'
fi

# ---- runtime ----
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export NCCL_TIMEOUT=7200 TORCH_NCCL_BLOCKING_WAIT=1
ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
EXPERIMENT_NAME=lastopd_${MODE}_${ACTOR_MODEL_NAME}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)
CKPT_PATH=$OUT_DIR/ckpt_${MODE}_${RUN_TAG}

if [ "${START_RAY:-1}" = "1" ]; then
  ray stop --force || true
  ray start --head --include-dashboard=false --disable-usage-stats
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct \
    algorithm.grpo_outcome_weight=1.0 \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$MINI_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS:-False} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF:-0.001} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +actor_rollout_ref.actor.use_rep_distillation=$USE_REP_DISTILLATION \
    +actor_rollout_ref.actor.rep_distillation_coef=$REP_DISTILLATION_COEF \
    +actor_rollout_ref.actor.rep_distillation_only=$REP_DISTILLATION_ONLY \
    +actor_rollout_ref.actor.rep_distillation_positions=$REP_DISTILLATION_POSITIONS \
    +actor_rollout_ref.actor.rep_distillation_last_k=$REP_DISTILLATION_LAST_K \
    +actor_rollout_ref.actor.rep_distillation_first_k=$REP_DISTILLATION_LAST_K \
    +actor_rollout_ref.actor.rep_distillation_layers=$REP_DISTILLATION_LAYERS \
    +actor_rollout_ref.actor.rep_projector_mode=$REP_PROJECTOR_MODE \
    +actor_rollout_ref.actor.rep_full_projector=$REP_FULL_PROJECTOR \
    +actor_rollout_ref.actor.rep_coef_decay_steps=$REP_COEF_DECAY_STEPS \
    +actor_rollout_ref.actor.token_coef_ramp_steps=$TOKEN_COEF_RAMP_STEPS \
    +actor_rollout_ref.actor.rep_sched_style=$REP_SCHED_STYLE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.6} \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=False \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=${REWARD_DTYPE:-$MODEL_DTYPE} \
    reward_model.micro_batch_size_per_gpu=${REWARD_MICRO_BS:-4} \
    reward_model.use_dynamic_bsz=False \
    custom_reward_function.path="verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=True \
    trainer.log_val_generations=2 \
    trainer.logger=['console','wandb'] \
    trainer.output_log_path=$OUT_DIR/logs/terminal/${EXPERIMENT_NAME}.log \
    trainer.project_name=LastOPD \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=$OUT_DIR/validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=1 \
    +trainer.total_training_steps=$TOTAL_STEPS \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=False ${EXTRA_HYDRA_ARGS:-}
