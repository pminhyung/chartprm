#!/usr/bin/env python3
"""Comprehensive analysis of v_hq-lite SFT training data."""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path("/ex_disk2/mhpark/poc/chartvr/data/sft_hq_lite_clean.jsonl")
COMPARISON_FILES = {
    "sft_hq_lite (pre-clean)": Path("/ex_disk2/mhpark/poc/chartvr/data/sft_hq_lite.jsonl"),
    "sft_30k": Path("/ex_disk2/mhpark/poc/chartvr/data/sft_30k.jsonl"),
    "sft_v9": Path("/ex_disk2/mhpark/poc/chartvr/data/sft_v9.jsonl"),
    "sft_v9_1": Path("/ex_disk2/mhpark/poc/chartvr/data/sft_v9_1.jsonl"),
    "grpo_v9": Path("/ex_disk2/mhpark/poc/chartvr/data/grpo_v9.jsonl"),
    "grpo_v9_1": Path("/ex_disk2/mhpark/poc/chartvr/data/grpo_v9_1.jsonl"),
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify_question(q):
    """Classify question type using keyword heuristics."""
    q_lower = q.lower().strip()

    # Yes/No
    if (q_lower.startswith(("is ", "are ", "does ", "do ", "did ", "was ", "were ",
                            "has ", "have ", "had ", "can ", "could ", "will ", "would ",
                            "should "))
        or q_lower.startswith("is it true")
        or "yes or no" in q_lower):
        return "Yes/No"

    # Count-based
    if any(w in q_lower for w in ["how many", "how much", "count of", "number of",
                                   "total number", "what is the total"]):
        return "Count/Quantity"

    # Comparison
    if any(w in q_lower for w in ["which is higher", "which is lower", "which is greater",
                                   "which is more", "which is less", "which has the highest",
                                   "which has the lowest", "compare", "difference between",
                                   "more than", "less than", "larger", "smaller",
                                   "which is the most", "which is the least",
                                   "which category", "which year", "which country",
                                   "which of the following", "rank"]):
        return "Comparison"

    # Trend/Pattern
    if any(w in q_lower for w in ["trend", "pattern", "increase", "decrease",
                                   "growth", "decline", "change over", "over time",
                                   "rising", "falling", "fluctuat"]):
        return "Trend/Pattern"

    # Reasoning/Inference
    if any(w in q_lower for w in ["why ", "why?", "what can be inferred",
                                   "what does this suggest", "explain",
                                   "what would happen", "what is the reason",
                                   "implication", "conclude", "based on",
                                   "what can you conclude", "interpret"]):
        return "Reasoning/Inference"

    # Simple numeric extraction
    if any(w in q_lower for w in ["what is the value", "what was the value",
                                   "what is the percentage", "what is the rate",
                                   "what is the amount", "what is the number",
                                   "what is the price", "what is the score",
                                   "approximate value", "estimated value",
                                   "what percent", "what fraction"]):
        return "Numeric Extraction"

    # General "What" questions (likely extraction)
    if q_lower.startswith("what "):
        return "What-question (general)"

    # "How" questions not caught above
    if q_lower.startswith("how "):
        return "How-question (general)"

    # Multichoice marker
    # (some questions may just be oddly phrased)
    return "Other"


def classify_answer(ans, answer_type_field=None):
    """Classify answer type."""
    ans_str = str(ans).strip()
    ans_lower = ans_str.lower()

    # Yes/No
    if ans_lower in ("yes", "no", "true", "false"):
        return "Yes/No"

    # Single letter (multichoice)
    if len(ans_str) == 1 and ans_str.upper() in "ABCDEFGH":
        return "Multiple Choice Letter"

    # Pure numeric
    try:
        # Remove commas, %, $
        cleaned = ans_str.replace(",", "").replace("%", "").replace("$", "").strip()
        float(cleaned)
        return "Numeric"
    except ValueError:
        pass

    # Short text vs long text
    words = ans_str.split()
    if len(words) <= 5:
        return "Short Text"
    elif len(words) <= 20:
        return "Medium Text"
    else:
        return "Long Text"


def answer_token_length(ans):
    """Approximate token count (whitespace split)."""
    return len(str(ans).split())


def extract_chart_type_from_source(source):
    """Try to extract chart type hint from source field."""
    return source if source else "unknown"


def print_table(title, counter, total=None):
    """Print a formatted frequency table."""
    if total is None:
        total = sum(counter.values())
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Category':<35} {'Count':>8} {'Pct':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8}")
    for key, count in counter.most_common():
        pct = count / total * 100 if total else 0
        print(f"  {str(key):<35} {count:>8,} {pct:>7.1f}%")
    print(f"  {'-'*35} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<35} {total:>8,}")


def print_histogram(title, values, bins=10):
    """Print an ASCII histogram."""
    if not values:
        return
    mn, mx = min(values), max(values)
    import math
    # Use quantile-based bins for better display
    values_sorted = sorted(values)
    n = len(values_sorted)

    # Fixed bins
    edges = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, max(mx + 1, 501)]

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  Mean: {sum(values)/len(values):.1f}, Median: {values_sorted[n//2]}, "
          f"Min: {mn}, Max: {mx}, Std: {(sum((v-sum(values)/n)**2 for v in values)/n)**0.5:.1f}")
    print()

    max_bar = 40
    bin_counts = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        cnt = sum(1 for v in values if lo <= v < hi)
        if cnt > 0:
            bin_counts.append((f"[{lo}-{hi})", cnt))

    max_cnt = max(c for _, c in bin_counts) if bin_counts else 1
    for label, cnt in bin_counts:
        bar_len = int(cnt / max_cnt * max_bar)
        pct = cnt / n * 100
        print(f"  {label:<12} {'#' * bar_len:<{max_bar}} {cnt:>7,} ({pct:>5.1f}%)")


def main():
    print("Loading data...")
    records = load_jsonl(DATA_PATH)
    N = len(records)
    print(f"Total records: {N:,}")

    # ---- Field presence ----
    all_keys = Counter()
    for r in records:
        for k in r.keys():
            all_keys[k] += 1
    print(f"\n{'='*70}")
    print(f"  FIELD PRESENCE")
    print(f"{'='*70}")
    for k, cnt in all_keys.most_common():
        print(f"  {k:<25} {cnt:>8,} / {N:,} ({cnt/N*100:.1f}%)")

    # ---- 1. Question Type Distribution ----
    q_types = Counter()
    for r in records:
        q_types[classify_question(r["question"])] += 1
    print_table("1. QUESTION TYPE DISTRIBUTION", q_types, N)

    # ---- 2. Answer Type Distribution ----
    a_types = Counter()
    for r in records:
        a_types[classify_answer(r["answer"], r.get("answer_type"))] += 1
    print_table("2a. ANSWER TYPE DISTRIBUTION (heuristic)", a_types, N)

    # Also show the raw answer_type field
    at_field = Counter()
    for r in records:
        at_field[r.get("answer_type", "N/A")] += 1
    print_table("2b. ANSWER TYPE FIELD (raw from data)", at_field, N)

    # ---- 3. Answer Length Distribution ----
    ans_lengths = [answer_token_length(r["answer"]) for r in records]
    print_histogram("3. ANSWER LENGTH (word count)", ans_lengths)

    # ---- 4. Source Distribution ----
    sources = Counter()
    for r in records:
        sources[r.get("source", "N/A")] += 1
    print_table("4a. SOURCE DISTRIBUTION", sources, N)

    origins = Counter()
    for r in records:
        origins[r.get("origin", "N/A")] += 1
    print_table("4b. ORIGIN DISTRIBUTION", origins, N)

    # ---- 5. Chart Type / Source breakdown ----
    # Since there's no explicit chart_type field, analyze the source field
    # and also look at image paths for clues
    img_dirs = Counter()
    for r in records:
        ip = r.get("image_path", "")
        # Extract directory structure
        parts = Path(ip).parts
        # Look for chart type hints in path
        if "charts_v9" in ip:
            # e.g., data/charts_v9/plotly_complex/png/...
            idx = parts.index("charts_v9") if "charts_v9" in parts else -1
            if idx >= 0 and idx + 1 < len(parts):
                img_dirs[parts[idx + 1]] += 1
            else:
                img_dirs["charts_v9/other"] += 1
        elif "chartqa" in ip.lower():
            img_dirs["chartqa"] += 1
        elif "charxiv" in ip.lower():
            img_dirs["charxiv"] += 1
        else:
            # First meaningful dir
            img_dirs["/".join(parts[-4:-2]) if len(parts) > 3 else ip[:50]] += 1
    print_table("5. IMAGE PATH DIRECTORY (chart type proxy)", img_dirs, N)

    # ---- 6. Difficulty Distribution ----
    diffs = Counter()
    for r in records:
        diffs[r.get("difficulty", "N/A")] += 1
    print_table("6. DIFFICULTY DISTRIBUTION", diffs, N)

    # ---- 7. Reasoning Steps Analysis ----
    has_reasoning = sum(1 for r in records if r.get("reasoning_steps"))
    reasoning_lengths = [len(r.get("reasoning_steps", "").split()) for r in records if r.get("reasoning_steps")]
    print(f"\n{'='*70}")
    print(f"  7. REASONING STEPS ANALYSIS")
    print(f"{'='*70}")
    print(f"  Records with reasoning_steps: {has_reasoning:,} / {N:,} ({has_reasoning/N*100:.1f}%)")
    if reasoning_lengths:
        print(f"  Reasoning word count — Mean: {sum(reasoning_lengths)/len(reasoning_lengths):.1f}, "
              f"Median: {sorted(reasoning_lengths)[len(reasoning_lengths)//2]}, "
              f"Min: {min(reasoning_lengths)}, Max: {max(reasoning_lengths)}")

    # Check for think tags
    think_tag_count = sum(1 for r in records
                          if "<think>" in str(r.get("reasoning_steps", ""))
                          or "<think>" in str(r.get("answer", "")))
    print(f"  Records with <think> tags: {think_tag_count:,}")

    # ---- 8. Conversation Format / Message Structure ----
    print(f"\n{'='*70}")
    print(f"  8. CONVERSATION FORMAT ANALYSIS")
    print(f"{'='*70}")
    # Check if any records have 'messages' or 'conversations' field
    has_messages = sum(1 for r in records if "messages" in r)
    has_conversations = sum(1 for r in records if "conversations" in r)
    print(f"  Records with 'messages' field: {has_messages:,}")
    print(f"  Records with 'conversations' field: {has_conversations:,}")
    print(f"  Format: raw Q/A fields (question, answer, reasoning_steps)")

    # Sample some questions and answers
    print(f"\n  Sample records (first 5):")
    for i, r in enumerate(records[:5]):
        q = r["question"][:80]
        a = str(r["answer"])[:60]
        print(f"    [{i}] Q: {q}...")
        print(f"        A: {a}")
        print(f"        type={r.get('answer_type')}, source={r.get('source')}, diff={r.get('difficulty')}")

    # ---- 9. Cross-tabulation: source x answer_type ----
    print(f"\n{'='*70}")
    print(f"  9. SOURCE x ANSWER_TYPE CROSS-TAB")
    print(f"{'='*70}")
    cross = defaultdict(Counter)
    for r in records:
        cross[r.get("source", "N/A")][r.get("answer_type", "N/A")] += 1

    all_atypes = sorted(set(r.get("answer_type", "N/A") for r in records))
    header = f"  {'Source':<25}" + "".join(f"{at:>12}" for at in all_atypes) + f"{'Total':>10}"
    print(header)
    print(f"  {'-'*25}" + "-" * (12 * len(all_atypes) + 10))
    for src in sorted(cross.keys()):
        row_total = sum(cross[src].values())
        row = f"  {src:<25}" + "".join(f"{cross[src].get(at, 0):>12}" for at in all_atypes) + f"{row_total:>10}"
        print(row)

    # ---- 10. Comparison with other datasets ----
    print(f"\n{'='*70}")
    print(f"  10. COMPARISON WITH OTHER SFT/GRPO DATASETS")
    print(f"{'='*70}")
    print(f"  {'Dataset':<30} {'Samples':>10} {'Fields'}")
    print(f"  {'-'*30} {'-'*10} {'-'*40}")
    print(f"  {'sft_hq_lite_clean':<30} {N:>10,} {list(records[0].keys())}")

    for name, path in COMPARISON_FILES.items():
        if path.exists():
            try:
                comp = load_jsonl(path)
                n_comp = len(comp)
                keys = list(comp[0].keys()) if comp else []
                print(f"  {name:<30} {n_comp:>10,} {keys}")

                # Show source distribution if available
                if comp and "source" in comp[0]:
                    src_c = Counter(r.get("source", "N/A") for r in comp)
                    top3 = src_c.most_common(3)
                    src_str = ", ".join(f"{k}={v}" for k, v in top3)
                    print(f"  {'  -> top sources':<30} {'':>10} {src_str}")
            except Exception as e:
                print(f"  {name:<30} ERROR: {e}")
        else:
            print(f"  {name:<30} {'(not found)':>10}")

    # ---- 11. Unique charts ----
    unique_charts = len(set(r.get("chart_slug", "") for r in records))
    unique_images = len(set(r.get("image_path", "") for r in records))
    print(f"\n{'='*70}")
    print(f"  11. UNIQUE CHART STATISTICS")
    print(f"{'='*70}")
    print(f"  Unique chart_slug values: {unique_charts:,}")
    print(f"  Unique image_path values: {unique_images:,}")
    print(f"  Avg QAs per chart: {N / unique_charts:.1f}" if unique_charts else "")

    # QAs per chart distribution
    chart_qa_counts = Counter(r.get("chart_slug", "") for r in records)
    qa_per_chart = Counter(v for v in chart_qa_counts.values())
    print(f"\n  QAs-per-chart distribution:")
    for n_qa in sorted(qa_per_chart.keys()):
        cnt = qa_per_chart[n_qa]
        print(f"    {n_qa} QA(s): {cnt:>6,} charts ({cnt/unique_charts*100:.1f}%)")

    # ---- 12. Multichoice option analysis ----
    mc_records = [r for r in records if r.get("answer_type") == "multichoice"]
    if mc_records:
        print(f"\n{'='*70}")
        print(f"  12. MULTICHOICE ANALYSIS ({len(mc_records):,} records)")
        print(f"{'='*70}")
        mc_answers = Counter(str(r["answer"]).strip().upper() for r in mc_records)
        print(f"  Answer letter distribution:")
        for letter, cnt in mc_answers.most_common():
            print(f"    {letter}: {cnt:>6,} ({cnt/len(mc_records)*100:.1f}%)")

    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
