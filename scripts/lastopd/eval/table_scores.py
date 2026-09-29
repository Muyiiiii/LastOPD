"""Paper-table scoring: per-sample rule grading (same grade_answer_verl as
grade_official.py) then avg@k / best@k with the paper's k per benchmark.
k <= generated n; the first k rollouts (file order) are used.
Usage: OUT_NAME=<run> python table_scores.py  -> $OUT_DIR/official_eval/<run>/table_scores.json
"""
import sys, os, json, signal
from pathlib import Path
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]          # repo root
OUT_ROOT = Path(os.environ.get("OUT_DIR", ROOT / "outputs"))
sys.path.insert(0, str(ROOT / "scripts/val/eval"))
from utils import grade_answer_verl

class _T(Exception): pass
def _alarm(s, f): raise _T()
signal.signal(signal.SIGALRM, _alarm)
def grade(resp, gt):
    signal.alarm(10)
    try: return bool(grade_answer_verl(resp, gt))
    except Exception: return False
    finally: signal.alarm(0)

# (file stem prefix, column name, k)
SPEC = [("math-500", "MATH-500", 8), ("aime24", "AIME24", 16), ("aime25", "AIME25", 16),
        ("amc23", "AMC23", 8), ("minerva", "Minerva", 4), ("olympiad-bench", "OlympiadBench", 4),
        ("aimo", "AIMO", 16), ("gsm8k", "GSM8K", 1)]
D = OUT_ROOT / "official_eval" / os.environ["OUT_NAME"]
out = {}
for stem, col, k in SPEC:
    fs = list(D.glob(f"{stem}_*.jsonl"))
    if not fs: out[col] = None; continue
    per = {}
    for line in open(fs[0]):
        d = json.loads(line); per.setdefault(int(d["example_id"]), []).append((d["response"], d["answer"]))
    avg, best = [], []
    for ex in sorted(per):
        s = [grade(r, g) for r, g in per[ex][:k]]
        avg.append(sum(s) / len(s)); best.append(max(s))
    out[col] = {"k": k, "n_avail": len(per[0]), "avg": round(100 * sum(avg) / len(avg), 2),
                "best": round(100 * sum(best) / len(best), 2), "num": len(per)}
    print(col, out[col], flush=True)
vals = [v for v in out.values() if v]
out["Mean"] = {"avg": round(sum(v["avg"] for v in vals) / len(vals), 2),
               "best": round(sum(v["best"] for v in vals) / len(vals), 2), "n_bench": len(vals)}
json.dump(out, open(D / "table_scores.json", "w"), indent=1)
print("MEAN", out["Mean"])
