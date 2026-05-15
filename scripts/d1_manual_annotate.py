"""D1 manual annotation — CLI annotator over first N samples of segmented traces.

For each sample, prints question + image path + thinking + each segment.
User scores each segment: coherent (Y/N), step_type, failure_mode.

Output: data/d1_pilot/annotations.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
IN_JSONL = BASE / "data/d1_pilot/segmented.jsonl"
OUT_JSONL = BASE / "data/d1_pilot/annotations.jsonl"

STEP_TYPES = ["value_ext", "comp", "arith", "cat", "trend", "plan", "backtrack", "concl", "other"]
FAILURE_MODES = ["none", "mid_split", "merged", "backtrack", "other"]


def load_done_ids() -> set[str]:
    if not OUT_JSONL.exists():
        return set()
    with open(OUT_JSONL) as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


def prompt(label: str, allowed: list[str] | None = None) -> str:
    while True:
        val = input(f"  {label}: ").strip()
        if not allowed or val in allowed:
            return val
        print(f"    invalid; allowed = {allowed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="Number of samples to annotate (default 20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reachqa_only", action="store_true",
                    help="Filter to ReachQA samples only (more multi-step)")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(IN_JSONL) if l.strip()]
    records = [r for r in records if r["n_steps"] > 0]
    if args.reachqa_only:
        records = [r for r in records if r["source"] == "reachqa_train"]

    import random
    rng = random.Random(args.seed)
    rng.shuffle(records)
    target = records[: args.n]
    done = load_done_ids()
    todo = [r for r in target if r["id"] not in done]

    print(f"\n[annotate] target={len(target)} done={len(done)} todo={len(todo)}\n")
    print(f"step_types  = {STEP_TYPES}")
    print(f"failure_modes = {FAILURE_MODES}")
    print(f"coherent: Y/N\n")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    for k, r in enumerate(todo):
        print("=" * 80)
        print(f"[{k+1}/{len(todo)}] id={r['id']} source={r['source']} n_steps={r['n_steps']}")
        print(f"image: {r['image_path']}")
        print(f"question: {r['question']}")
        print(f"gold: {r['gold_answer'][:200]}{'...' if len(r['gold_answer'])>200 else ''}")
        print(f"\n--- THINKING ({len(r['thinking'])} chars) ---\n{r['thinking']}\n")

        seg_anns = []
        for i, step in enumerate(r["steps"]):
            print(f"\n--- segment {i+1}/{len(r['steps'])} ({r['step_token_lens'][i]} tokens) ---")
            print(step)
            print()
            coherent = prompt("coherent (Y/N)", ["Y", "N", "y", "n"]).upper()
            step_type = prompt("step_type", STEP_TYPES)
            fail = "none" if coherent == "Y" else prompt("failure_mode", FAILURE_MODES)
            seg_anns.append({"idx": i, "coherent": coherent, "step_type": step_type, "failure": fail})

        rec_out = {"id": r["id"], "source": r["source"], "n_steps": r["n_steps"], "segments": seg_anns}
        with open(OUT_JSONL, "a") as f:
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
        print(f"[saved] -> {OUT_JSONL}")

    print(f"\n[done] total annotated: {len(target)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
