"""
ChartVCR Reward Scoring Dashboard

Reasoning Chain 분해 → 문장별 평가 → Causal Attribution → 최종 Reward 계산 과정을
단계별로 시각화하는 Streamlit 대시보드.

실행: streamlit run scripts/reward_scoring_dashboard.py --server.port 8503
"""
import streamlit as st
import json
import os
import re
import math
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

# ═══════════════════════════════════════════════════
# Reward functions (from train_grpo.py)
# ═══════════════════════════════════════════════════

NUM_RE = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'
CALC_RE = rf'({NUM_RE})\s*([+\-*/×÷])\s*({NUM_RE})\s*[=≈]\s*({NUM_RE})'


def get_table_values(csv_path):
    try:
        df = pd.read_csv(csv_path)
        vals = set()
        for col in df.columns:
            for v in df[col]:
                try:
                    vals.add(float(v))
                except:
                    pass
        return vals, df
    except:
        return set(), None


def gaussian_score(model_val, table_val, sigma=0.10):
    if table_val == 0:
        return 1.0 if abs(model_val) < 0.01 else 0.0
    return math.exp(-0.5 * (abs(model_val - table_val) / (abs(table_val) * sigma)) ** 2)


def analyze_sentence(sent, table_vals, tainted, sigma=0.10):
    """Analyze a single sentence and return detailed scoring."""
    nums = []
    for m in re.findall(NUM_RE, sent):
        try:
            nums.append(float(m.replace(',', '')))
        except:
            pass

    result = {
        "text": sent,
        "numbers": nums,
        "has_numbers": len(nums) > 0,
        "input_quality": 1.0,
        "logic_score": 1.0,
        "sentence_reward": 0.0,
        "label": "skip",
        "details": [],
    }

    if not nums:
        result["label"] = "no_numbers"
        return result

    # Input quality
    iq_list = []
    uses_tainted = False
    for n in nums:
        if any(abs(n - t) / max(abs(t), 1e-10) < 0.05 for t in tainted):
            uses_tainted = True
            iq_list.append(0.3)
            result["details"].append(f"  ⚠️ {n} is TAINTED (from prior error)")
        else:
            best_score = 0.0
            best_match = None
            for tv in table_vals:
                s = gaussian_score(n, tv, sigma)
                if s > best_score:
                    best_score = s
                    best_match = tv
            iq_list.append(best_score)
            if best_score > 0.7:
                result["details"].append(f"  ✅ {n} matches table value {best_match} (score={best_score:.3f})")
            elif best_score > 0.3:
                result["details"].append(f"  ⚠️ {n} ~partial match~ {best_match} (score={best_score:.3f})")
            else:
                result["details"].append(f"  ❌ {n} NOT in table (best={best_match}, score={best_score:.3f})")

    result["input_quality"] = min(iq_list) if iq_list else 1.0

    # Logic score (arithmetic check)
    cm = re.search(CALC_RE, sent)
    if cm:
        try:
            a = float(cm.group(1).replace(',', ''))
            op = cm.group(2).replace('×', '*').replace('÷', '/')
            b = float(cm.group(3).replace(',', ''))
            stated = float(cm.group(4).replace(',', ''))
            if op in '+-*/' and (op != '/' or b != 0):
                expected = eval(f"{a}{op}{b}")
                if abs(stated - expected) / max(abs(expected), 1e-10) < 0.05:
                    result["logic_score"] = 1.0
                    result["details"].append(f"  ✅ Arithmetic correct: {a}{op}{b} = {stated} (expected {expected:.4f})")
                else:
                    result["logic_score"] = 0.0
                    result["details"].append(f"  ❌ Arithmetic WRONG: {a}{op}{b} = {stated} (expected {expected:.4f})")
        except:
            result["logic_score"] = 1.0
    else:
        result["details"].append(f"  ℹ️ No explicit arithmetic found")

    # Sentence reward
    result["sentence_reward"] = result["logic_score"] * result["input_quality"]

    # Label
    if result["sentence_reward"] < 0.5 and not uses_tainted:
        result["label"] = "source_error"
    elif result["sentence_reward"] < 0.5 and uses_tainted:
        result["label"] = "propagated_error"
    else:
        result["label"] = "correct"

    return result


def full_reward_analysis(response, gold_answer, csv_path="", sigma=0.10):
    """Complete ChartVCR reward analysis pipeline."""
    # 1. Extract answer
    ans_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    extracted_answer = ans_match.group(1).strip() if ans_match else ""
    if not extracted_answer:
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        extracted_answer = lines[-1] if lines else ""

    # 2. Accuracy reward
    def relaxed_match(pred, gold):
        p = re.sub(r'[,%$]', '', pred.strip())
        g = re.sub(r'[,%$]', '', gold.strip())
        try:
            pf, gf = float(p), float(g)
            if gf == 0:
                return abs(pf) < 0.01
            return abs(pf - gf) / abs(gf) <= 0.05
        except:
            return p.lower() == g.lower()

    r_acc = 1.0 if relaxed_match(extracted_answer, gold_answer) else 0.0

    # 3. Format reward
    r_fmt = 0.0
    has_think = '<think>' in response.lower() or 'let' in response.lower()[:50]
    if has_think:
        r_fmt += 0.5
        sentences = [s.strip() for s in response.split('\n') if s.strip() and len(s.strip()) > 5]
        if len(sentences) >= 3:
            r_fmt += 0.5
    r_fmt = min(r_fmt, 1.0)

    # 4. Process reward (sentence-by-sentence)
    table_vals = set()
    table_df = None
    if csv_path and os.path.exists(csv_path):
        table_vals, table_df = get_table_values(csv_path)

    # Split reasoning into sentences
    sentences = [s.strip() for s in response.split('\n') if s.strip() and len(s.strip()) > 5]

    tainted = set()
    sentence_analyses = []
    for sent in sentences:
        analysis = analyze_sentence(sent, table_vals, tainted, sigma)
        sentence_analyses.append(analysis)
        # Taint numbers from source errors
        if analysis["label"] == "source_error":
            for n in analysis["numbers"]:
                tainted.add(n)

    # Aggregate process reward
    valid_rewards = [a["sentence_reward"] for a in sentence_analyses if a["has_numbers"]]
    r_proc = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0

    # 5. Total reward
    w_acc, w_proc, w_fmt = 0.5, 0.3, 0.2
    total = w_acc * r_acc + w_proc * r_proc + w_fmt * r_fmt

    return {
        "extracted_answer": extracted_answer,
        "gold_answer": gold_answer,
        "r_accuracy": r_acc,
        "r_process": r_proc,
        "r_format": r_fmt,
        "total_reward": total,
        "weights": {"accuracy": w_acc, "process": w_proc, "format": w_fmt},
        "sentence_analyses": sentence_analyses,
        "table_values": sorted(table_vals)[:20] if table_vals else [],
        "table_df": table_df,
        "tainted_numbers": sorted(tainted),
        "num_sentences": len(sentences),
        "num_verified": len(valid_rewards),
    }


# ═══════════════════════════════════════════════════
# Streamlit Dashboard
# ═══════════════════════════════════════════════════

st.set_page_config(page_title="ChartVCR Reward Scoring", layout="wide")
st.title("🔬 ChartVCR Reward Scoring Dashboard")
st.markdown("Reasoning Chain 분해 → 문장별 검증 → Causal Attribution → 최종 Reward")

# Load data
@st.cache_data
def load_eval_data():
    results = []
    eval_file = os.path.join(BASE, "results/v4/qwen3vl_zeroshot_chartqa_human.json")
    if os.path.exists(eval_file):
        with open(eval_file) as f:
            data = json.load(f)
        for r in data["results"]:
            results.append(r)
    return results

@st.cache_data
def load_grpo_data():
    grpo_file = os.path.join(BASE, "data/grpo_train.json")
    if os.path.exists(grpo_file):
        with open(grpo_file) as f:
            return json.load(f)
    return []

eval_data = load_eval_data()
grpo_data = load_grpo_data()

# Build lookup for CSV paths
csv_lookup = {}
for item in grpo_data:
    q = item.get("problem", "")
    csv_lookup[q] = item.get("csv_path", "")

# Sidebar
st.sidebar.header("Settings")
sigma = st.sidebar.slider("σ (Gaussian tolerance)", 0.01, 0.30, 0.10, 0.01)
filter_type = st.sidebar.radio("Filter", ["All", "Correct only", "Wrong only", "Has computation"])

# Filter samples
filtered = eval_data
if filter_type == "Correct only":
    filtered = [r for r in eval_data if r["accuracy"] > 0]
elif filter_type == "Wrong only":
    filtered = [r for r in eval_data if r["accuracy"] == 0]
elif filter_type == "Has computation":
    filtered = [r for r in eval_data if re.search(CALC_RE, r["response"])]

st.sidebar.metric("Total samples", len(eval_data))
st.sidebar.metric("Filtered", len(filtered))

if not filtered:
    st.warning("No samples found")
    st.stop()

# Sample selector
idx = st.sidebar.number_input("Sample index", 0, len(filtered) - 1, 0)
sample = filtered[idx]

# Find CSV path
csv_path = csv_lookup.get(sample["question"], "")

# Run analysis
analysis = full_reward_analysis(sample["response"], sample["gold_answer"], csv_path, sigma)

# ═══════════════════════════════════════════════════
# Main display
# ═══════════════════════════════════════════════════

# Header
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Reward", f"{analysis['total_reward']:.3f}")
with col2:
    color = "🟢" if analysis["r_accuracy"] > 0 else "🔴"
    st.metric(f"R_accuracy {color}", f"{analysis['r_accuracy']:.1f}")
with col3:
    st.metric("R_process", f"{analysis['r_process']:.3f}")
with col4:
    st.metric("R_format", f"{analysis['r_format']:.1f}")

# Formula
st.info(
    f"**Total = {analysis['weights']['accuracy']}×R_acc + {analysis['weights']['process']}×R_proc + {analysis['weights']['format']}×R_fmt**  \n"
    f"= {analysis['weights']['accuracy']}×{analysis['r_accuracy']:.1f} + "
    f"{analysis['weights']['process']}×{analysis['r_process']:.3f} + "
    f"{analysis['weights']['format']}×{analysis['r_format']:.1f} = **{analysis['total_reward']:.3f}**"
)

# Question & Answer
st.subheader("📋 Question & Answer")
st.markdown(f"**Question:** {sample['question']}")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**Gold Answer:** `{sample['gold_answer']}`")
with col_b:
    match = "✅ CORRECT" if analysis["r_accuracy"] > 0 else "❌ WRONG"
    st.markdown(f"**Extracted:** `{analysis['extracted_answer']}` → {match}")

# Data Table (if available)
if analysis["table_df"] is not None:
    with st.expander("📊 Data Table (CSV)", expanded=False):
        st.dataframe(analysis["table_df"], use_container_width=True)
        st.caption(f"Table values used for verification: {analysis['table_values'][:15]}...")

# ═══════════════════════════════════════════════════
# Reasoning Chain Analysis
# ═══════════════════════════════════════════════════

st.subheader("🔗 Reasoning Chain — Sentence-by-Sentence Analysis")

for i, sa in enumerate(analysis["sentence_analyses"]):
    if not sa["has_numbers"] and sa["label"] == "no_numbers":
        # Skip non-numeric sentences (show collapsed)
        with st.expander(f"Step {i+1}: 💬 Text-only (no numbers)", expanded=False):
            st.text(sa["text"][:200])
        continue

    # Determine icon
    if sa["label"] == "correct":
        icon = "✅"
        border_color = "green"
    elif sa["label"] == "source_error":
        icon = "❌"
        border_color = "red"
    elif sa["label"] == "propagated_error":
        icon = "🔶"
        border_color = "orange"
    else:
        icon = "⬜"
        border_color = "gray"

    with st.expander(
        f"Step {i+1}: {icon} {sa['label'].upper()} — "
        f"logic={sa['logic_score']:.1f} × input_q={sa['input_quality']:.3f} = **{sa['sentence_reward']:.3f}**",
        expanded=True
    ):
        st.markdown(f"> {sa['text']}")

        if sa["numbers"]:
            st.markdown(f"**Numbers found:** {sa['numbers']}")

        for detail in sa["details"]:
            st.markdown(detail)

        # Score bar
        cols = st.columns(3)
        with cols[0]:
            st.progress(sa["logic_score"], text=f"Logic: {sa['logic_score']:.2f}")
        with cols[1]:
            st.progress(min(sa["input_quality"], 1.0), text=f"Input Quality: {sa['input_quality']:.3f}")
        with cols[2]:
            st.progress(min(sa["sentence_reward"], 1.0), text=f"Sentence Reward: {sa['sentence_reward']:.3f}")

# Tainted numbers
if analysis["tainted_numbers"]:
    st.warning(f"🔶 **Tainted numbers** (from source errors, propagated to downstream): {analysis['tainted_numbers']}")

# ═══════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════

st.subheader("📊 Reward Breakdown Summary")

summary_data = {
    "Component": ["R_accuracy", "R_process", "R_format", "**TOTAL**"],
    "Score": [analysis["r_accuracy"], analysis["r_process"], analysis["r_format"], analysis["total_reward"]],
    "Weight": [0.5, 0.3, 0.2, 1.0],
    "Weighted": [
        0.5 * analysis["r_accuracy"],
        0.3 * analysis["r_process"],
        0.2 * analysis["r_format"],
        analysis["total_reward"],
    ],
}
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.caption(
    f"Sentences: {analysis['num_sentences']} total, "
    f"{analysis['num_verified']} with numbers (verified), "
    f"σ={sigma}"
)

# Full response
with st.expander("📝 Full Model Response", expanded=False):
    st.text(sample["response"])
