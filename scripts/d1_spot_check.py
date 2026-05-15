"""D1 spot check — 5 random samples, ENTER = ok, type 'BAD' = broken.

Guide §2.5: 10-min manual safety net for auto-validation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
DEFAULT_IN = BASE / "data/d1_pilot/segmented_v2.jsonl"
REPORT_JSON = BASE / "data/d1_pilot/spot_check.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    data = [json.loads(l) for l in open(args.input) if l.strip()]
    data = [d for d in data if d.get("n_steps", 0) > 0]
    rng = random.Random(args.seed)
    samples = rng.sample(data, args.n)

    print(f"=== {args.n}-sample spot check ===")
    print(f"For each segment, press ENTER if reasonable, type 'BAD' (or 'b') if obviously broken.\n")

    results = []
    for k, s in enumerate(samples):
        print("=" * 80)
        print(f"[{k+1}/{args.n}] id={s['id']} source={s['source']} variant={s.get('sampling_variant','?')} n_steps={s['n_steps']}")
        print(f"image: {s['image_path']}")
        print(f"Q: {s['question']}")
        print(f"gold: {s['gold_answer'][:160]}{'...' if len(s['gold_answer'])>160 else ''}")
        bad_segs = []
        for i, step in enumerate(s["steps"]):
            print(f"\n--- segment {i+1}/{s['n_steps']} ({s['step_token_lens'][i]} tok) ---")
            print(step)
            v = input("  ENTER=ok / BAD: ").strip().upper()
            if v in ("BAD", "B"):
                bad_segs.append(i)
        sample_verdict = "BAD" if bad_segs else "OK"
        results.append({"id": s["id"], "verdict": sample_verdict, "bad_segments": bad_segs})
        print(f"  → sample verdict: {sample_verdict}")

    n_bad = sum(1 for r in results if r["verdict"] == "BAD")
    print(f"\n=== SUMMARY ===")
    print(f"BAD samples: {n_bad}/{args.n}")
    for r in results:
        print(f"  {r['id']}: {r['verdict']} (bad segments: {r['bad_segments']})")

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps({"results": results, "n_bad": n_bad, "n_total": args.n}, indent=2))
    print(f"\n[saved] {REPORT_JSON}")


if __name__ == "__main__":
    sys.exit(main() or 0)
