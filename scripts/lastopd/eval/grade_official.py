"""Rule-based grading of the official-protocol generations (same grade_answer_verl as OPRD scripts/val/eval).
Env: OUT_NAME, MODEL_PATH (tokenizer for length stats). Writes grading_results.json next to the jsonl files.
"""
import sys, os, signal
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]          # repo root
OUT_ROOT = Path(os.environ.get("OUT_DIR", ROOT / "outputs"))
sys.path.insert(0, str(ROOT / "scripts/val/eval"))
from utils import grade_answer_verl

# Per-sample hard timeout (08-01): a degenerate m4s2 answer (huge-exponent
# expression) wedged sympy/mpmath in an effectively unbounded bignum
# computation — 7.5 h at 100% CPU on ONE sample. Anything sympy cannot verify
# within GRADE_TIMEOUT_S is graded incorrect; such outputs are degenerate,
# not near-misses, so this does not change any legitimate score.
GRADE_TIMEOUT_S = int(os.environ.get("GRADE_TIMEOUT_S", "10"))
_timeouts = 0

class _GradeTimeout(Exception):
    pass

def _alarm(signum, frame):
    raise _GradeTimeout()

signal.signal(signal.SIGALRM, _alarm)

def grade_guarded(response, gt):
    global _timeouts
    signal.alarm(GRADE_TIMEOUT_S)
    try:
        return grade_answer_verl(response, gt)
    except _GradeTimeout:
        _timeouts += 1
        return False
    except Exception:
        return False
    finally:
        signal.alarm(0)
from transformers import AutoTokenizer
import json
import re
from pathlib import Path

OUT_NAME = os.environ["OUT_NAME"]
EVAL_DIR = OUT_ROOT / "official_eval" / OUT_NAME
OUTPUT_FILE = EVAL_DIR / "grading_results.json"
LEN_TOK_PATH = os.environ["MODEL_PATH"]

length_tokenizer = AutoTokenizer.from_pretrained(LEN_TOK_PATH, local_files_only=True, trust_remote_code=True)

def get_len(seq):
    return len(length_tokenizer.encode(seq)) if length_tokenizer else len(seq)

def get_diverse_score(sequences, n=4):
    distinct_ngrams = set()
    total_ngrams = 0
    for seq in sequences:
        tokens = seq.split()
        for i in range(len(tokens) - n + 1):
            distinct_ngrams.add(tuple(tokens[i:i + n]))
            total_ngrams += 1
    return len(distinct_ngrams) / total_ngrams if total_ngrams > 0 else 0

def process_jsonl_file(file_name):
    results = []
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            id = int(data["example_id"])
            while len(results) <= id:
                results.append({"gt": None, "responses": []})
            results[id]["gt"] = data["answer"]
            results[id]["responses"].append(data["response"])
    return results

def parse_hyperparameters_from_filename(filename):
    match = re.search(r"_t(?P<temperature>[\d.]+)_p(?P<top_p>[\d.]+)_n(?P<n>\d+)-MNT(?P<max_tokens>\d+)", filename)
    return match.groupdict() if match else {}

def grade_file(file_path):
    hyperparams = parse_hyperparameters_from_filename(file_path.name)
    if not hyperparams:
        print(f"Skipping file with unrecognized format: {file_path}")
        return None
    hyperparams["task_name"] = file_path.stem.split("_")[0]

    df = process_jsonl_file(file_path)
    num_pred = len(df[0]["responses"])

    diverse, response_lengths, rule_based_scores = [], [], []
    without_boxed = 0
    for i in range(len(df)):
        responses_list = [str(r) for r in df[i]["responses"]]
        gt = df[i]["gt"]
        response_lengths += [get_len(r) for r in responses_list]
        without_boxed += sum("boxed" not in r for r in responses_list)
        for response in responses_list:
            rule_based_scores.append(grade_guarded(response, gt))
        diverse.append(get_diverse_score(responses_list))

    final_scores = [bool(s) for s in rule_based_scores]
    avg_scores = [sum(final_scores[i:i + num_pred]) / num_pred for i in range(0, len(final_scores), num_pred)]
    best = [max(final_scores[i:i + num_pred]) for i in range(0, len(final_scores), num_pred)]

    return {
        "hyperparameters": hyperparams,
        "mean_score": sum(avg_scores) / len(avg_scores),
        "distinct_4gram": sum(diverse) / len(diverse),
        "best_score": sum(best) / len(best),
        "solve_none": sum(1 for a in avg_scores if a == 0),
        "solve_all": sum(1 for a in avg_scores if a == 1),
        "avg_output_length": sum(response_lengths) / len(response_lengths),
        "format_error_rollouts": without_boxed,
        "grade_timeouts": _timeouts,
    }

def main():
    all_results = []
    if not EVAL_DIR.exists():
        print(f"Directory {EVAL_DIR} does not exist.")
        return
    for file_path in sorted(EVAL_DIR.glob("*.jsonl")):
        print(f"Processing file: {file_path}")
        r = grade_file(file_path)
        if r:
            all_results.append(r)
            print(json.dumps(r, indent=2))
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Grading results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
