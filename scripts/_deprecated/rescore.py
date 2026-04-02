"""
Re-score all rows with improved answer extraction (v2).
Reads existing JSONL eval files, re-extracts answers, re-scores accuracy.
"""
import json
import re
import os

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

ROWS = {
    "row1_zeroshot":  "results/v6/qwen35_4b_zeroshot",
    "row2_baseline":  "results/v6/baseline_v6",
    "row3_outcome":   "results/v6b/row3_v6b",
    "row4_cvr_v1":    "results/v6b/cvr_v6b",
    "row5_cvr_v3":    "results/v6b/cvr_v3",
}

BENCHMARKS = ["chartqa_human", "chartqa_augmented", "charxiv_reasoning", "chartqa_pro", "chartmuseum"]


def extract_answer_v2(content: str, reasoning_content: str = "") -> str:
    """Improved answer extraction applied uniformly to all rows."""
    # Reconstruct full response
    response = f"<think>{reasoning_content}</think>\n{content}" if reasoning_content else content

    # Priority 1: <answer> tag in content
    m = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        # Strip units/qualifiers
        ans = re.sub(r'\b(million|billion|trillion|thousand)\b', '', ans, flags=re.I).strip()
        ans = re.sub(r'\b(percent|percentage)\b', '%', ans, flags=re.I).strip()
        ans = re.sub(r'\b(dollars|USD|EUR|GBP)\b', '', ans, flags=re.I).strip()
        ans = re.sub(r'^(approximately|about|around|roughly|~)\s*', '', ans, flags=re.I).strip()
        ans = re.sub(r'\s+', ' ', ans).strip()
        # If still too long or markdown bullet → extract first number
        if len(ans) > 50 or ans.startswith('*') or ans.startswith('-'):
            nums = re.findall(r'[-+]?\d+(?:[,\.]\d+)*', ans)
            if nums:
                return nums[0].replace(',', '')
        return ans

    # Priority 2: <answer> tag in full response
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Priority 3: After </think>
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        if after:
            lines = [l.strip() for l in after.split('\n') if l.strip()]
            if lines:
                candidate = lines[0]
                if len(candidate) < 80:
                    return candidate
                nums = re.findall(r'[-+]?\d+(?:[,\.]\d+)*', candidate)
                if nums:
                    return nums[0].replace(',', '')

    # Priority 4: truncation — last number in reasoning
    nums = re.findall(r'[-+]?\d+(?:[,\.]\d+)*', response[-500:])
    if nums:
        return nums[-1].replace(',', '')

    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def relaxed_accuracy(pred: str, gold: str) -> float:
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return 1.0 if abs(pf) < 0.01 else 0.0
        return 1.0 if abs(pf - gf) / abs(gf) <= 0.05 else 0.0
    except ValueError:
        return 1.0 if p.lower() == g.lower() else 0.0


def rescore_file(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        return None

    results = []
    original_acc = []
    new_acc = []

    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            original_acc.append(d.get("accuracy", 0))

            content = d.get("content", "") or ""
            reasoning = d.get("reasoning_content", "") or ""
            gold = d.get("gold_answer", "")

            new_pred = extract_answer_v2(content, reasoning)
            new_accuracy = relaxed_accuracy(new_pred, gold)
            new_acc.append(new_accuracy)

            d["predicted_answer_v2"] = new_pred
            d["accuracy_v2"] = new_accuracy
            results.append(d)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    orig = sum(original_acc) / len(original_acc) * 100
    new = sum(new_acc) / len(new_acc) * 100
    return orig, new, len(results)


def main():
    output_base = os.path.join(BASE, "results/rescored")

    print(f"\n{'Model':<22} {'Benchmark':<22} {'Old':>7} {'New':>7} {'Δ':>6} {'N':>6}")
    print("=" * 72)

    summary = {}  # row -> bench -> (old, new)

    for row_name, row_dir in ROWS.items():
        summary[row_name] = {}
        for bench in BENCHMARKS:
            input_path = os.path.join(BASE, row_dir, f"{bench}.jsonl")
            output_path = os.path.join(output_base, row_name, f"{bench}.jsonl")

            result = rescore_file(input_path, output_path)
            if result is None:
                summary[row_name][bench] = None
                continue

            orig, new, n = result
            delta = new - orig
            summary[row_name][bench] = (orig, new)
            sign = "+" if delta >= 0 else ""
            print(f"{row_name:<22} {bench:<22} {orig:>6.1f}% {new:>6.1f}% {sign}{delta:>5.1f}% {n:>6}")

    # Summary table
    bench_short = ["CQA-H", "CQA-A", "CharXiv", "CQA-Pro", "ChartMus"]
    bench_keys  = ["chartqa_human", "chartqa_augmented", "charxiv_reasoning", "chartqa_pro", "chartmuseum"]

    print(f"\n\n{'='*80}")
    print("FAIR COMPARISON TABLE (extract_v2 applied to ALL rows)")
    print(f"{'='*80}")
    print(f"\n{'Model':<22}", end="")
    for b in bench_short:
        print(f"  {b:>10}", end="")
    print()
    print("-" * 78)

    for row_name in ROWS:
        row_label = {
            "row1_zeroshot":  "Row1 Zero-shot",
            "row2_baseline":  "Row2 ChartQA+outcome",
            "row3_outcome":   "Row3 our+outcome",
            "row4_cvr_v1":    "Row4 CVR V1",
            "row5_cvr_v3":    "Row5 CVR V3",
        }[row_name]
        print(f"{row_label:<22}", end="")
        for bk in bench_keys:
            v = summary[row_name].get(bk)
            if v:
                print(f"  {v[1]:>9.1f}%", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()

    print(f"\n{'Old extraction (original)':}")
    print(f"{'Model':<22}", end="")
    for b in bench_short:
        print(f"  {b:>10}", end="")
    print()
    print("-" * 78)
    for row_name in ROWS:
        row_label = {
            "row1_zeroshot":  "Row1 Zero-shot",
            "row2_baseline":  "Row2 ChartQA+outcome",
            "row3_outcome":   "Row3 our+outcome",
            "row4_cvr_v1":    "Row4 CVR V1",
            "row5_cvr_v3":    "Row5 CVR V3",
        }[row_name]
        print(f"{row_label:<22}", end="")
        for bk in bench_keys:
            v = summary[row_name].get(bk)
            if v:
                print(f"  {v[0]:>9.1f}%", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()


if __name__ == "__main__":
    main()
