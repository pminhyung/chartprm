"""D2 — prepare zero-shot 4B baseline by filtering archived d1_pilot_4b traces.

Filters data/d1_pilot_4b/segmented_v3.jsonl to D2 sample ids.
Adds 'raw_response' = (content + reasoning_content) for downstream uniformity.
"""
import json, os, sys
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))


def main():
    src = BASE / "data/d1_pilot_4b/segmented_v3.jsonl"
    ids_src = BASE / "data/d2_pilot/samples.jsonl"
    dst = BASE / "data/d2_pilot/baseline_zeroshot_4b_segmented.jsonl"

    if not src.exists():
        print(f"FATAL: missing {src}")
        return 1

    d2_ids = {json.loads(l)["id"] for l in open(ids_src) if l.strip()}
    src_recs = {json.loads(l)["id"]: json.loads(l) for l in open(src) if l.strip()}
    missing = d2_ids - set(src_recs.keys())
    print(f"d2_ids={len(d2_ids)}  src={len(src_recs)}  missing={len(missing)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dst, "w") as f:
        for sid in d2_ids:
            r = src_recs.get(sid)
            if r is None:
                continue
            # raw_response = final answer-bearing text. Use content if present, else reasoning_content.
            raw = r.get("content") or r.get("reasoning_content") or ""
            r["raw_response"] = raw
            r["baseline_model"] = "zeroshot_4b"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} -> {dst}")


if __name__ == "__main__":
    sys.exit(main() or 0)
