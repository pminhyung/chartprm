"""
ChartVCR Reward Scoring Dashboard (v5 — audited logic)

Reasoning Chain 분해 → 문장별 평가 → Causal Attribution → 최종 Reward 계산 과정을
단계별로 시각화. reasoning_chain.py v5 로직 사용.

실행: streamlit run scripts/reward_scoring_dashboard.py --server.port 8503
"""
import streamlit as st
import json
import os
import re
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

# v5 audited logic
from code.rewards.reasoning_chain import (
    compute_process_reward,
    split_reasoning,
    extract_chart_numbers,
    best_table_match,
)

# ═══════════════════════════════════════════════════
# Full reward analysis using v5 logic
# ═══════════════════════════════════════════════════

def full_reward_analysis(response, gold_answer, csv_path="", sigma=0.10):
    """Complete ChartVCR reward with v5 reasoning_chain."""
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
            if gf == 0: return abs(pf) < 0.01
            return abs(pf - gf) / abs(gf) <= 0.05
        except:
            return p.lower() == g.lower()

    r_acc = 1.0 if relaxed_match(extracted_answer, gold_answer) else 0.0

    # 3. Format reward
    r_fmt = 0.0
    has_reasoning = len(response) > 100
    if has_reasoning:
        r_fmt += 0.5
        sentences = split_reasoning(response)
        if len(sentences) >= 3:
            r_fmt += 0.5
    r_fmt = min(r_fmt, 1.0)

    # 4. Process reward (v5 audited logic)
    r_proc, sentence_analyses = compute_process_reward(response, csv_path, sigma)

    # Load table for display
    table_df = None
    table_vals_list = []
    if csv_path and os.path.exists(csv_path):
        try:
            table_df = pd.read_csv(csv_path)
            for col in table_df.columns:
                for v in table_df[col]:
                    try: table_vals_list.append(float(v))
                    except: pass
        except: pass

    # 5. Total reward
    w_acc, w_proc, w_fmt = 0.5, 0.3, 0.2
    total = w_acc * r_acc + w_proc * r_proc + w_fmt * r_fmt

    # Collect tainted
    tainted = set()
    for a in sentence_analyses:
        if a.label == "source_error":
            for n in a.chart_numbers:
                tainted.add(n)

    return {
        "extracted_answer": extracted_answer,
        "gold_answer": gold_answer,
        "r_accuracy": r_acc,
        "r_process": r_proc,
        "r_format": r_fmt,
        "total_reward": total,
        "weights": {"accuracy": w_acc, "process": w_proc, "format": w_fmt},
        "sentence_analyses": sentence_analyses,
        "table_values": sorted(set(table_vals_list))[:20],
        "table_df": table_df,
        "tainted_numbers": sorted(tainted),
        "num_sentences": len(sentence_analyses),
        "num_verified": sum(1 for a in sentence_analyses if a.label != "skip"),
    }


# ═══════════════════════════════════════════════════
# Streamlit Dashboard
# ═══════════════════════════════════════════════════

st.set_page_config(page_title="ChartVCR Reward Scoring", layout="wide")
st.title("🔬 ChartVCR Reward Scoring Dashboard (v5)")
st.markdown("Reasoning Chain 분해 → 문장별 검증 → Causal Attribution → 최종 Reward")

_CALC_RE = r'\d+\.?\d*\s*[+\-*/×÷]\s*\d+\.?\d*\s*[=≈]\s*\d+\.?\d*'

@st.cache_data
def load_eval_data():
    eval_file = os.path.join(BASE, "results/v4/qwen3vl_zeroshot_chartqa_human.json")
    if os.path.exists(eval_file):
        with open(eval_file) as f:
            return json.load(f)["results"]
    return []

@st.cache_data
def load_grpo_data():
    grpo_file = os.path.join(BASE, "data/grpo_train.json")
    if os.path.exists(grpo_file):
        with open(grpo_file) as f:
            return json.load(f)
    return []

eval_data = load_eval_data()
grpo_data = load_grpo_data()

csv_lookup = {}
img_lookup = {}
for item in grpo_data:
    q = item.get("problem", "")
    csv_lookup[q] = item.get("csv_path", "")
    img_lookup[q] = item.get("image", "")

# Sidebar
st.sidebar.header("Settings")
sigma = st.sidebar.slider("σ (Gaussian tolerance)", 0.01, 0.30, 0.10, 0.01)
filter_type = st.sidebar.radio("Filter", [
    "All", "Correct only", "Wrong only", "Has computation", "Has CSV"
])

filtered = eval_data
if filter_type == "Correct only":
    filtered = [r for r in eval_data if r["accuracy"] > 0]
elif filter_type == "Wrong only":
    filtered = [r for r in eval_data if r["accuracy"] == 0]
elif filter_type == "Has computation":
    filtered = [r for r in eval_data if re.search(_CALC_RE, r["response"])]
elif filter_type == "Has CSV":
    filtered = [r for r in eval_data if csv_lookup.get(r["question"], "") and
                os.path.exists(csv_lookup.get(r["question"], ""))]

st.sidebar.metric("Total samples", len(eval_data))
st.sidebar.metric("Filtered", len(filtered))

if not filtered:
    st.warning("No samples found")
    st.stop()

# Navigation
col_nav1, col_nav2, col_nav3 = st.sidebar.columns(3)
if "idx" not in st.session_state:
    st.session_state.idx = 0
with col_nav1:
    if st.button("⬅ Prev"):
        st.session_state.idx = max(0, st.session_state.idx - 1)
with col_nav3:
    if st.button("Next ➡"):
        st.session_state.idx = min(len(filtered) - 1, st.session_state.idx + 1)
with col_nav2:
    st.session_state.idx = st.number_input("Index", 0, len(filtered)-1, st.session_state.idx, label_visibility="collapsed")

sample = filtered[st.session_state.idx]
csv_path = csv_lookup.get(sample["question"], "")
img_path = img_lookup.get(sample["question"], "")

# Run analysis
analysis = full_reward_analysis(sample["response"], sample["gold_answer"], csv_path, sigma)

# ═══════════════════════════════════════════════════
# Header metrics
# ═══════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Reward", f"{analysis['total_reward']:.3f}")
with col2:
    icon = "🟢" if analysis["r_accuracy"] > 0 else "🔴"
    st.metric(f"R_accuracy {icon}", f"{analysis['r_accuracy']:.1f}")
with col3:
    st.metric("R_process", f"{analysis['r_process']:.3f}")
with col4:
    st.metric("R_format", f"{analysis['r_format']:.1f}")

st.info(
    f"**Total = {analysis['weights']['accuracy']}×R_acc + "
    f"{analysis['weights']['process']}×R_proc + "
    f"{analysis['weights']['format']}×R_fmt** = "
    f"{analysis['weights']['accuracy']}×{analysis['r_accuracy']:.1f} + "
    f"{analysis['weights']['process']}×{analysis['r_process']:.3f} + "
    f"{analysis['weights']['format']}×{analysis['r_format']:.1f} = "
    f"**{analysis['total_reward']:.3f}**"
)

# ═══════════════════════════════════════════════════
# Question + Chart Image
# ═══════════════════════════════════════════════════
st.subheader("📋 Question & Answer")
col_img, col_qa = st.columns([1, 1])
with col_img:
    if img_path and os.path.exists(img_path):
        st.image(img_path, caption="Chart Image", use_container_width=True)
    else:
        st.info("Chart image not available (test set — no matching train image)")
with col_qa:
    st.markdown(f"**Question:** {sample['question']}")
    st.markdown(f"**Gold Answer:** `{sample['gold_answer']}`")
    match = "✅ CORRECT" if analysis["r_accuracy"] > 0 else "❌ WRONG"
    st.markdown(f"**Extracted:** `{analysis['extracted_answer']}` → {match}")
    st.markdown(f"**Model:** Qwen3-VL-8B-Thinking (zero-shot)")
    has_csv = "✅" if csv_path and os.path.exists(csv_path) else "❌"
    st.markdown(f"**CSV available:** {has_csv}")

# Data Table
if analysis["table_df"] is not None:
    with st.expander("📊 Data Table (CSV)", expanded=False):
        st.dataframe(analysis["table_df"], use_container_width=True)
        st.caption(f"Table values: {analysis['table_values'][:15]}...")

# ═══════════════════════════════════════════════════
# Reasoning Chain (v5 logic)
# ═══════════════════════════════════════════════════
st.subheader("🔗 Reasoning Chain — Sentence-by-Sentence (v5)")
st.caption(f"{analysis['num_sentences']} sentences, {analysis['num_verified']} verified, σ={sigma}")

for sa in analysis["sentence_analyses"]:
    if sa.label == "skip":
        with st.expander(f"Step {sa.index+1}: 💬 Text-only (skipped)", expanded=False):
            st.text(sa.text[:200])
        continue

    icons = {"correct": "✅", "source_error": "❌", "propagated_error": "🔶"}
    icon = icons.get(sa.label, "⬜")

    with st.expander(
        f"Step {sa.index+1}: {icon} {sa.label.upper()} — "
        f"logic={sa.logic_score:.1f} × iq={sa.input_quality:.3f} = **{sa.sentence_reward:.3f}**",
        expanded=(sa.label != "skip")
    ):
        st.markdown(f"> {sa.text}")

        # Show numbers: chart vs filtered
        if sa.chart_numbers:
            st.markdown(f"**Chart numbers:** {sa.chart_numbers}")
        if sa.filtered_numbers:
            st.caption(f"Filtered out: {sa.filtered_numbers} (years/indices/counts)")

        # Details
        for d in sa.number_details:
            st.markdown(d)
        if sa.arithmetic:
            st.markdown(f"**Arithmetic:** {sa.arithmetic}")

        # Score bars
        cols = st.columns(3)
        with cols[0]:
            st.progress(sa.logic_score, text=f"Logic: {sa.logic_score:.2f}")
        with cols[1]:
            st.progress(min(sa.input_quality, 1.0), text=f"Input Quality: {sa.input_quality:.3f}")
        with cols[2]:
            st.progress(min(sa.sentence_reward, 1.0), text=f"Reward: {sa.sentence_reward:.3f}")

# Tainted
if analysis["tainted_numbers"]:
    st.warning(f"🔶 **Tainted numbers** (propagated from source errors): {analysis['tainted_numbers']}")

# ═══════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════
st.subheader("📊 Reward Breakdown")
summary = pd.DataFrame({
    "Component": ["R_accuracy", "R_process", "R_format", "**TOTAL**"],
    "Score": [analysis["r_accuracy"], analysis["r_process"], analysis["r_format"], analysis["total_reward"]],
    "Weight": [0.5, 0.3, 0.2, 1.0],
    "Weighted": [
        0.5 * analysis["r_accuracy"],
        0.3 * analysis["r_process"],
        0.2 * analysis["r_format"],
        analysis["total_reward"],
    ],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

# Full response
with st.expander("📝 Full Model Response", expanded=False):
    st.text(sample["response"])
