"""Official-protocol generation for the 8 benchmarks (vLLM, T=0.7, top_p=0.95, max 31744 tokens).
Env: MODEL_PATH (HF model dir), OUT_NAME (run name), NUM_GPUS or GPU_IDS, *_N per-benchmark sample counts
(MATH500_N AIMO_N AMC23_N GSM8K_N MINERVA_N OLYMPIAD_N), TASKS_FILTER. Output: $OUT_DIR/official_eval/<OUT_NAME>/<task>.jsonl
"""
import os
import json
import re
import argparse
import concurrent.futures
import multiprocessing
import gc
import torch
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    destroy_model_parallel = None
try:
    from vllm.distributed.parallel_state import destroy_distributed_environment
except ImportError:
    destroy_distributed_environment = None

import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]          # repo root
OUT_ROOT = Path(os.environ.get("OUT_DIR", ROOT / "outputs"))
DATA_DIR = str(ROOT / "scripts/val/data")
TEST_DATA_DIR = DATA_DIR

MODEL_NAMES = [os.environ["MODEL_PATH"]]
OUT_NAME = os.environ["OUT_NAME"]

TASKS = [
    {"name": "AIME24", "path": f"{DATA_DIR}/AIME24/test.parquet", "N": 16},
    {"name": "AIME25", "path": f"{DATA_DIR}/AIME25/test.parquet", "N": 16},
]
if int(os.environ.get("MATH500_N", "0")) > 0:
    TASKS.append({"name": "MATH-500", "path": f"{DATA_DIR}/MATH-500/test.parquet",
                  "N": int(os.environ.get("MATH500_N", "0"))})
if int(os.environ.get("AIMO_N", "0")) > 0:
    TASKS.append({"name": "AIMO", "path": f"{DATA_DIR}/AIMO/test.parquet",
                  "N": int(os.environ.get("AIMO_N", "0"))})
if int(os.environ.get("AMC23_N", "0")) > 0:
    TASKS.append({"name": "AMC23", "path": f"{DATA_DIR}/AMC23/test.parquet",
                  "N": int(os.environ.get("AMC23_N", "0"))})
if int(os.environ.get("GSM8K_N", "0")) > 0:
    TASKS.append({"name": "GSM8K", "path": f"{DATA_DIR}/GSM8K/test.parquet",
                  "N": int(os.environ.get("GSM8K_N", "0"))})
_TASKS_FILTER = {s.strip() for s in os.environ.get("TASKS_FILTER", "").split(",") if s.strip()}
if int(os.environ.get("MINERVA_N", "0")) > 0 or "Minerva" in _TASKS_FILTER:
    TASKS.append({"name": "Minerva", "path": f"{TEST_DATA_DIR}/Minerva/test.parquet",
                  "N": int(os.environ.get("MINERVA_N", "4"))})
if int(os.environ.get("OLYMPIAD_N", "0")) > 0 or "Olympiad-Bench" in _TASKS_FILTER:
    TASKS.append({"name": "Olympiad-Bench", "path": f"{TEST_DATA_DIR}/Olympiad-Bench/test.parquet",
                  "N": int(os.environ.get("OLYMPIAD_N", "4"))})
if os.environ.get("AIMO_ONLY", "0") == "1":
    TASKS = [t for t in TASKS if t["name"] == "AIMO"]
if _TASKS_FILTER:
    TASKS = [t for t in TASKS if t["name"] in _TASKS_FILTER]

BOXED_SUFFIX = os.environ.get("BOXED_SUFFIX", "True").strip().lower() not in ("0", "false")
if BOXED_SUFFIX:
    PROMPT_TEMPLATE = """{problem} Please reason step by step, and put your final answer within \\boxed{{}}."""
else:
    PROMPT_TEMPLATE = """{problem}"""
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", "31744"))
REP_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.0"))
TEMPERATURE = 0.7
TOP_P       = 0.95
REPLACE     = False


def load_samples(filepath: str):
    df = pd.read_parquet(filepath)
    samples = [
        {
            "example_id": i,
            "prompt": df.at[i, "prompt"][0]["content"].strip(),
            "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
        }
        for i in range(len(df))
    ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def split_rollout_ids(rollout_ids, num_workers):
    chunks = [[] for _ in range(num_workers)]
    for idx, rollout_id in enumerate(rollout_ids):
        chunks[idx % num_workers].append(rollout_id)
    return chunks


def worker_process(args_tuple):
    model_name, samples, rollout_id_list, gpu_id, enable_thinking = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # 单机 4 worker 同时起 vLLM 引擎会抢同一个分布式初始化端口(EADDRINUSE),
    # 按 GPU 号错开固定端口段避免竞争
    os.environ["VLLM_PORT"] = str(36000 + 200 * int(gpu_id))
    results = []
    llm = None
    stop_token_ids = []
    try:
        print(
            f"[GPU {gpu_id}] | Model: {model_name} | rollouts len={len(rollout_id_list)} "
            f"| loading model (TP=1, enable_thinking={enable_thinking})...",
            flush=True,
        )
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
        )
        try:
            tokenizer = llm.get_tokenizer()
            for stop_token in ["<|im_end|>", "<|endoftext|>"]:
                try:
                    if hasattr(tokenizer, "encode"):
                        encoded = tokenizer.encode(stop_token, add_special_tokens=False)
                        # 仅接受单 token(真正的特殊符);多 token 说明该词表无此特殊符,跳过
                        if encoded and len(encoded) == 1:
                            stop_token_ids.append(encoded[0])
                except Exception:
                    pass
            print(f"[GPU {gpu_id}] stop_token_ids={stop_token_ids} (eos={tokenizer.eos_token_id} 由 vLLM 默认处理)", flush=True)
        except Exception as e:
            tokenizer = None
            print(f"[GPU {gpu_id}] Warning: Could not get tokenizer for stop tokens: {e}", flush=True)

        for rollout_id in rollout_id_list:
            sampling = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                repetition_penalty=REP_PENALTY,
                stop_token_ids=stop_token_ids if stop_token_ids else None,
            )
            if tokenizer is None:
                raise RuntimeError("Tokenizer is required for apply_chat_template, but it could not be loaded.")
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": s["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for s in samples
            ]
            outputs = llm.generate(formatted_prompts, sampling, use_tqdm=False)
            for sample, out in zip(samples, outputs):
                results.append(
                    {
                        "example_id": sample["example_id"],
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "seed": rollout_id,
                        "response": out.outputs[0].text,
                    }
                )
            print(f"[GPU {gpu_id}] rollout {rollout_id} done ({len(samples)} prompts)", flush=True)
    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error: {e}", flush=True)
    finally:
        print(f"[GPU {gpu_id}] Cleaning up resources...", flush=True)
        if llm is not None:
            del llm
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass
        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[GPU {gpu_id}] Cleanup done.", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation rollouts with vLLM (official protocol).")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking_group.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    # 默认值可由环境变量 ENABLE_THINKING("True"/"False")覆盖;CLI 旗标优先级更高。
    # 环境变量不设时默认 False,与原版行为一致。
    parser.set_defaults(
        enable_thinking=os.environ.get("ENABLE_THINKING", "False").strip().lower() in ("1", "true"))
    args = parser.parse_args()

    # GPU_IDS="4,5,6,7" pins workers to physical GPUs (a 4-GPU slot inside the
    # shared workspace); otherwise the first NUM_GPUS devices are used.
    if os.environ.get("GPU_IDS", "").strip():
        gpu_workers = [g.strip() for g in os.environ["GPU_IDS"].split(",") if g.strip()]
    else:
        gpu_workers = [str(g) for g in range(int(os.environ.get("NUM_GPUS", "4")))]
    num_workers = len(gpu_workers)
    print(f"GPU workers (one model per GPU): {gpu_workers}")

    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}\nStarting evaluation for model: {model_name}\n{'='*50}")
        OUT_DIR = OUT_ROOT / "official_eval" / OUT_NAME
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        for task in TASKS:
            task_name, task_path, N = task["name"], task["path"], task["N"]
            print(f"Starting evaluation for task: {task_name} (N={N})")
            out_path = OUT_DIR / f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{N}-MNT{MAX_TOKENS}.jsonl"
            if not REPLACE and out_path.exists():
                print(f"Result file already exists at '{out_path}'. Skipping.")
                continue
            samples = load_samples(task_path)
            for sample in samples:
                sample["prompt"] = PROMPT_TEMPLATE.format(problem=sample["prompt"])
            if samples:
                print("Example prompt after formatting:")
                print(samples[0]["prompt"])
            rollout_ids = list(range(N))
            rollout_chunks = split_rollout_ids(rollout_ids, num_workers)
            all_results = []
            args_list = [
                (model_name, samples, rollout_chunks[i], gpu_workers[i], args.enable_thinking)
                for i in range(num_workers)
            ]
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
                futures = [ex.submit(worker_process, tup) for tup in args_list]
                for fut in tqdm(concurrent.futures.as_completed(futures),
                                total=len(futures), desc=f"GPU workers ({task_name})"):
                    try:
                        all_results.extend(fut.result())
                    except Exception as e:
                        print(f"A worker process failed with error: {e}")
            print(f"Total generations collected for {task_name}: {len(all_results)}")
            if all_results:
                with out_path.open("w", encoding="utf-8") as f:
                    for item in all_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"Saved results for {task_name} to {out_path}")
            else:
                print(f"No results collected for {task_name} (Check for errors).")

    print("===== OFFICIAL EVAL GEN DONE =====")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
