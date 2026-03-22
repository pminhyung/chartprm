"""
ChartVCR Verifier Reward Dashboard (v6)

Qwen3.5 structured verifier vs rule-based(v5) reward scoring 비교.
Sample-by-sample: LLM extracted values, table matching, causal attribution, scores.

실행: streamlit run scripts/verifier_reward_dashboard.py --server.port 8504
"""
import streamlit as st
import json
import os
import pandas as pd

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")
DATA_PATH = os.path.join(BASE, "results/v4/train_reward_30samples.jsonl")

st.set_page_config(page_title="ChartVCR Verifier Reward", layout="wide")
st.title("ChartVCR Verifier Reward Dashboard (v6)")
st.markdown("Qwen3.5 Structured Extraction + Deterministic Scoring vs Rule-based (v5)")


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


data = load_data()
if not data:
    st.error(f"No data found at {DATA_PATH}")
    st.stop()

# ═══════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════
st.sidebar.header("Navigation")
filter_type = st.sidebar.radio("Filter", [
    "All", "Correct only", "Wrong only",
    "V6 > V5 (LLM better)", "V5 > V6 (Rule better)",
    "V6 has extractions",
])

filtered = data
if filter_type == "Correct only":
    filtered = [d for d in data if d["accuracy"] > 0]
elif filter_type == "Wrong only":
    filtered = [d for d in data if d["accuracy"] == 0]
elif filter_type == "V6 > V5 (LLM better)":
    filtered = [d for d in data if d["r_proc_v6"] > d["r_proc_v5"] + 0.01]
elif filter_type == "V5 > V6 (Rule better)":
    filtered = [d for d in data if d["r_proc_v5"] > d["r_proc_v6"] + 0.01]
elif filter_type == "V6 has extractions":
    filtered = [d for d in data if len(d.get("v6_extractions", [])) > 0]

st.sidebar.metric("Total", len(data))
st.sidebar.metric("Filtered", len(filtered))

if not filtered:
    st.warning("No samples match filter")
    st.stop()

# Navigation
if "idx" not in st.session_state:
    st.session_state.idx = 0
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("Prev"):
        st.session_state.idx = max(0, st.session_state.idx - 1)
with col3:
    if st.button("Next"):
        st.session_state.idx = min(len(filtered) - 1, st.session_state.idx + 1)
with col2:
    st.session_state.idx = st.number_input(
        "Idx", 0, len(filtered) - 1, st.session_state.idx,
        label_visibility="collapsed"
    )

sample = filtered[st.session_state.idx]

# ═══════════════════════════════════════════
# Aggregate Summary
# ═══════════════════════════════════════════
with st.expander("Summary (all samples)", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    accs = [d["accuracy"] for d in data]
    v5s = [d["r_proc_v5"] for d in data]
    v6s = [d["r_proc_v6"] for d in data]
    t5s = [d["total_reward_v5"] for d in data]
    t6s = [d["total_reward_v6"] for d in data]

    with col_a:
        st.metric("Accuracy", f"{sum(accs)/len(accs):.1%}")
    with col_b:
        st.metric("Avg R_proc v5", f"{sum(v5s)/len(v5s):.3f}")
    with col_c:
        st.metric("Avg R_proc v6", f"{sum(v6s)/len(v6s):.3f}")
    with col_d:
        v6_better = sum(1 for v5, v6 in zip(v5s, v6s) if v6 > v5 + 0.01)
        st.metric("V6 > V5", f"{v6_better}/{len(data)}")

    # Comparison table
    comp_df = pd.DataFrame({
        "Sample": [d["sample_id"] for d in data],
        "Question": [d["question"][:40] for d in data],
        "Acc": [d["accuracy"] for d in data],
        "V5": [d["r_proc_v5"] for d in data],
        "V6": [d["r_proc_v6"] for d in data],
        "Diff": [round(d["r_proc_v6"] - d["r_proc_v5"], 3) for d in data],
        "Total_V5": [d["total_reward_v5"] for d in data],
        "Total_V6": [d["total_reward_v6"] for d in data],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# Header Metrics
# ═══════════════════════════════════════════
st.divider()
st.markdown(f"### Sample #{sample['sample_id']}")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    icon = "O" if sample["accuracy"] > 0 else "X"
    st.metric(f"Accuracy ({icon})", f"{sample['accuracy']:.0f}")
with col2:
    st.metric("R_proc V5", f"{sample['r_proc_v5']:.3f}")
with col3:
    st.metric("R_proc V6", f"{sample['r_proc_v6']:.3f}")
with col4:
    diff = sample["r_proc_v6"] - sample["r_proc_v5"]
    st.metric("V6-V5 Diff", f"{diff:+.3f}")
with col5:
    st.metric("Total V6", f"{sample['total_reward_v6']:.3f}")

# ═══════════════════════════════════════════
# Question & Chart Image
# ═══════════════════════════════════════════
st.subheader("Question & Answer")

img_path = sample.get("image_path", "")
if img_path and not os.path.exists(img_path):
    img_path = ""

col_img, col_qa = st.columns([1, 1])
with col_img:
    if img_path:
        st.image(img_path, caption="Chart Image", use_container_width=True)
    else:
        st.info("Chart image not available")
with col_qa:
    st.markdown(f"**Q:** {sample['question']}")
    match_icon = "O CORRECT" if sample["accuracy"] > 0 else "X WRONG"
    st.markdown(f"**Gold:** `{sample['gold_answer']}` | **Pred:** `{sample['predicted_answer']}` -> {match_icon}")
    st.markdown(f"**Sentences:** {sample['num_sentences']} | **Time:** {sample['inference_time_s']}s")
    if sample.get("table_values"):
        st.caption(f"Table values: {sample['table_values']}")

# ═══════════════════════════════════════════
# V6 LLM Extractions (main content)
# ═══════════════════════════════════════════
st.subheader("V6: Qwen3.5 Structured Extractions")

v6_exts = sample.get("v6_extractions", [])
v6_labels = sample.get("v6_labels", {})

if v6_exts:
    st.caption(
        f"Verified: {len(v6_exts)} sentences | "
        f"Labels: {v6_labels}"
    )

    for ext in v6_exts:
        icons = {"correct": "O", "source_error": "X", "propagated_error": "~"}
        icon = icons.get(ext["label"], "?")

        with st.expander(
            f"[{icon}] {ext['label'].upper()} | "
            f"iq={ext['input_quality']:.3f} logic={ext['logic_score']:.1f} "
            f"r={ext['sentence_reward']:.3f}",
            expanded=(ext["label"] != "correct"),
        ):
            st.markdown(f"> {ext['text']}")

            # LLM extracted values
            if ext["llm_values"]:
                st.markdown("**LLM Extracted Values:**")
                for entity, val, unit in ext["llm_values"]:
                    unit_str = f" {unit}" if unit else ""
                    st.markdown(f"- `{entity}`: **{val}**{unit_str}")

            # LLM extracted computations
            if ext["llm_comps"]:
                st.markdown("**LLM Extracted Computations:**")
                for expr, result in ext["llm_comps"]:
                    st.markdown(f"- `{expr}` = {result}")

            # Table matching scores
            if ext["value_scores"]:
                st.markdown("**Table Matching:**")
                for val, score, matched in ext["value_scores"]:
                    if score > 0.7:
                        badge = "MATCH"
                    elif score > 0.3:
                        badge = "PARTIAL"
                    else:
                        badge = "MISS"
                    st.markdown(f"- {val} -> {matched} (score={score:.3f}) [{badge}]")

            # Computation verification
            if ext["comp_results"]:
                for expr, correct in ext["comp_results"]:
                    result = "CORRECT" if correct else "WRONG"
                    st.markdown(f"**Arithmetic:** [{result}] {expr}")

            # Score bars
            cols = st.columns(3)
            with cols[0]:
                st.progress(min(ext["input_quality"], 1.0), text=f"IQ: {ext['input_quality']:.3f}")
            with cols[1]:
                st.progress(ext["logic_score"], text=f"Logic: {ext['logic_score']:.1f}")
            with cols[2]:
                st.progress(min(ext["sentence_reward"], 1.0), text=f"Reward: {ext['sentence_reward']:.3f}")

            # Verifier raw response
            if ext.get("verifier_raw"):
                st.caption("Verifier raw:")
                try:
                    import json as _json
                    pretty = _json.dumps(_json.loads(ext["verifier_raw"]), indent=2, ensure_ascii=False)
                except Exception:
                    pretty = ext["verifier_raw"]
                st.code(pretty, language="json")
else:
    st.info("No verifiable sentences extracted by V6 (all skip)")

# ═══════════════════════════════════════════
# V5 vs V6 Comparison
# ═══════════════════════════════════════════
st.subheader("V5 vs V6 Label Distribution")
v5_labels = sample.get("v5_labels", {})
col_v5, col_v6 = st.columns(2)
with col_v5:
    st.markdown("**V5 (Rule-based)**")
    for label, count in sorted(v5_labels.items(), key=lambda x: -x[1]):
        st.markdown(f"- {label}: {count}")
    st.metric("R_proc", f"{sample['r_proc_v5']:.3f}")
with col_v6:
    st.markdown("**V6 (LLM Structured)**")
    for label, count in sorted(v6_labels.items(), key=lambda x: -x[1]):
        st.markdown(f"- {label}: {count}")
    st.metric("R_proc", f"{sample['r_proc_v6']:.3f}")

# ═══════════════════════════════════════════
# Reward Breakdown
# ═══════════════════════════════════════════
st.subheader("Reward Breakdown")
breakdown = pd.DataFrame({
    "Component": ["R_accuracy", "R_process", "R_format", "TOTAL"],
    "Weight": [0.5, 0.3, 0.2, 1.0],
    "V5 Score": [
        sample["r_accuracy"], sample["r_proc_v5"], sample["r_format"],
        sample["total_reward_v5"],
    ],
    "V6 Score": [
        sample["r_accuracy"], sample["r_proc_v6"], sample["r_format"],
        sample["total_reward_v6"],
    ],
    "V5 Weighted": [
        0.5 * sample["r_accuracy"], 0.3 * sample["r_proc_v5"],
        0.2 * sample["r_format"], sample["total_reward_v5"],
    ],
    "V6 Weighted": [
        0.5 * sample["r_accuracy"], 0.3 * sample["r_proc_v6"],
        0.2 * sample["r_format"], sample["total_reward_v6"],
    ],
})
st.dataframe(breakdown, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# Full Response
# ═══════════════════════════════════════════
with st.expander("Full Model Response", expanded=False):
    st.text(sample["response"])
