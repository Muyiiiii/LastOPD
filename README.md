# LastOPD: Last-Layer On-Policy Representation Distillation with Crossfade

LastOPD distills a teacher LLM into a student **on the student's own rollouts** with two signals that are
crossfaded over training:

- **Latent loss** (early): normalized MSE between the student's **last-layer** hidden states, mapped through a
  trainable MLP, and the teacher's last-layer hidden states on **all response tokens**.
- **Token loss** (late): reverse-KL on the student's top-*k* tokens (on-policy distillation, OPD).

The latent weight decays 1 → 0 and the token weight ramps 0 → 1 over the first `N` steps
(`REP_COEF_DECAY_STEPS` / `TOKEN_COEF_RAMP_STEPS`, linear or hard).
Setting both to 0 gives **LastOPD-always** (both terms on for the whole run).

The code is a small patch (≈260 lines in `verl/`) on top of
[OPRD](https://github.com/ShenzhiYang2000/OPRD) (branch `OPRD-Bridge`, commit `93816fda`); see
[`UPSTREAM.md`](UPSTREAM.md). OPRD's own README is kept at [`docs/README_OPRD.md`](docs/README_OPRD.md);
its OPRD-Vanilla / OPRD-Bridge configurations still work unchanged.

## What the patch adds

| File | Change |
|---|---|
| `verl/verl/utils/rep_distillation.py` | `REP_PROJECTOR_MODE=full` with an MLP projector (student → teacher space); last-layer / all-position extraction |
| `verl/verl/workers/actor/dp_actor.py` | latent loss wiring; **crossfade schedule** (`rep_coef_decay_steps`, `token_coef_ramp_steps`, `rep_sched_style`) |
| `verl/verl/workers/config/actor.py` | the three schedule fields + projector options |
| `verl/verl/workers/fsdp_workers.py` | teacher last-layer hidden extraction; optional bf16 host cache (`REP_TEACHER_CACHE_DTYPE=bf16`) |
| `verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py`, `trainer/ppo/ray_trainer.py` | projector state in checkpoints; plumbing |
| `scripts/lastopd/` | training launcher, paper cells, official 8-benchmark evaluation |
| `scripts/val/data/GSM8K/` | GSM8K test split in the OPRD parquet format (`openai/gsm8k`, `main`) |

## Setup

Same environment as OPRD / verl (torch 2.8, vLLM 0.11, flash-attn, ray). Follow `docs/README_OPRD.md` for the
environment and for downloading `dapo-math-17k.parquet` and the `test_data/<bench>/test.parquet` files; the
in-tree copies under `scripts/val/data/` are used by the official evaluation.

## Train

```bash
export MODEL_DIR=/path/to/models      # HF folders: DeepSeek-R1-Distill-Qwen-1.5B, JustRL-DeepSeek-1.5B, Qwen3-1.7B-Base, Qwen3-4B, Qwen3-8B
export DATA_DIR=/path/to/data         # dapo-math-17k.parquet, test_data/MATH-500/test.parquet, ...
export OUT_DIR=/path/to/outputs
bash scripts/lastopd/cells.sh lastopd            # ours (crossfade 10/10), 1.5B -> 1.5B
bash scripts/lastopd/cells.sh lastopd_always     # no crossfade
bash scripts/lastopd/cells.sh vanilla            # OPRD-Vanilla baseline (all layers, last 250 tokens, latent only)
bash scripts/lastopd/cells.sh q4_lastopd         # Qwen3-4B -> Qwen3-1.7B-Base
```

`scripts/lastopd/train.sh` is the single launcher; every knob is an environment variable with the paper default
(`REP_DISTILLATION_COEF=1000`, `REP_DISTILLATION_LAYERS=last`, `REP_DISTILLATION_POSITIONS=all`,
`REP_FULL_PROJECTOR=mlp`, crossfade 10/10, 62 steps, mini-batch 32 × 4 responses, 8k tokens, constant lr 1e-5,
KL off, fp32). In-training validation is MATH-500 (mean@8, T=0.7) every 10 steps.

## Evaluate (official protocol, 8 benchmarks)

```bash
bash scripts/lastopd/eval.sh $OUT_DIR/ckpt_combined_lastopd/global_step_62/actor lastopd
# -> $OUT_DIR/official_eval/lastopd/{table_scores.json, grading_results.json, *.jsonl}
```

Merges the FSDP checkpoint to HF, generates with vLLM (T=0.7, top_p=0.95, 31744 tokens), grades with the OPRD rule
grader, and reports avg@k / best@k with the paper's k: MATH-500 @8, AIME24/25 @16, AMC23 @8, Minerva @4,
OlympiadBench @4, AIMO @16, GSM8K acc@1.

## Loss scale

The latent loss is `2(1 − cos)/D` after L2 normalization (OPRD's `normalized_mse_loss`), so with `D = 1536` a
weight of 1000 puts it at ≈1.3·(1 − cos) — the same order as the token loss (|pg_loss| ≈ 0.05 at step 1).
With weight 1 (OPRD's default) the latent term is ~1800× smaller than the token term.

## Citation

If you use this code, please also cite OPRD and OPD (see `docs/README_OPRD.md`).

## License

The verl code base is Apache-2.0 (`verl/LICENSE`) and the LastOPD additions are released under the same
license. The upstream OPRD repository does not declare a license for its own modifications; please refer to
the OPRD authors for the terms covering that part of the code.
