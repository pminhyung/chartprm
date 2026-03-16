"""
GRPO Training Data Inspection Dashboard
========================================
Browse and inspect GRPO training samples one-by-one.

Launch:
    streamlit run scripts/grpo_data_inspection_dashboard.py --server.port 8501
"""

import json
import os
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "/ex_disk2/mhpark/poc/chartvr/data/grpo_train.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data


def is_numeric_answer(answer: str) -> bool:
    try:
        float(answer.replace(",", ""))
        return True
    except (ValueError, AttributeError):
        return False


def answer_type(answer: str) -> str:
    return "numeric" if is_numeric_answer(answer) else "text"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GRPO Train Data Inspector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GRPO Training Data Inspector")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
all_data = load_data()

# ---------------------------------------------------------------------------
# Sidebar -- filters and statistics
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

# Split file filter
split_files = sorted(set(s["split_file"] for s in all_data))
selected_splits = st.sidebar.multiselect(
    "Split file",
    options=split_files,
    default=split_files,
)

# Source filter
sources = sorted(set(s["source"] for s in all_data))
selected_sources = st.sidebar.multiselect(
    "Source",
    options=sources,
    default=sources,
)

# Answer type filter
selected_answer_type = st.sidebar.radio(
    "Answer type",
    options=["All", "numeric", "text"],
    index=0,
)

# Apply filters
filtered_data = [
    s for s in all_data
    if s["split_file"] in selected_splits
    and s["source"] in selected_sources
    and (selected_answer_type == "All" or answer_type(s["solution"]) == selected_answer_type)
]

# ---------------------------------------------------------------------------
# Sidebar -- statistics
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Statistics")
st.sidebar.metric("Total samples", f"{len(all_data):,}")
st.sidebar.metric("Filtered samples", f"{len(filtered_data):,}")

st.sidebar.subheader("By split_file")
for sf in split_files:
    cnt = sum(1 for s in all_data if s["split_file"] == sf)
    st.sidebar.text(f"  {sf}: {cnt:,}")

st.sidebar.subheader("By answer type")
num_count = sum(1 for s in all_data if is_numeric_answer(s["solution"]))
txt_count = len(all_data) - num_count
st.sidebar.text(f"  numeric: {num_count:,}")
st.sidebar.text(f"  text: {txt_count:,}")

# ---------------------------------------------------------------------------
# Main area -- navigator
# ---------------------------------------------------------------------------
if len(filtered_data) == 0:
    st.warning("No samples match the current filters.")
    st.stop()

# Session state for current index
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

# Clamp index to valid range
max_idx = len(filtered_data) - 1
st.session_state.sample_idx = min(st.session_state.sample_idx, max_idx)
st.session_state.sample_idx = max(st.session_state.sample_idx, 0)

st.markdown("### Sample Navigator")
nav_cols = st.columns([1, 1, 2, 3])

with nav_cols[0]:
    if st.button("Prev", use_container_width=True):
        st.session_state.sample_idx = max(0, st.session_state.sample_idx - 1)

with nav_cols[1]:
    if st.button("Next", use_container_width=True):
        st.session_state.sample_idx = min(max_idx, st.session_state.sample_idx + 1)

with nav_cols[2]:
    jump_idx = st.number_input(
        "Go to index",
        min_value=0,
        max_value=max_idx,
        value=st.session_state.sample_idx,
        step=1,
        key="jump_input",
    )
    if jump_idx != st.session_state.sample_idx:
        st.session_state.sample_idx = jump_idx

with nav_cols[3]:
    st.markdown(
        f"**Sample {st.session_state.sample_idx + 1} / {len(filtered_data):,}**"
    )

# ---------------------------------------------------------------------------
# Current sample
# ---------------------------------------------------------------------------
sample = filtered_data[st.session_state.sample_idx]

st.markdown("---")

# Metadata row
meta_cols = st.columns(4)
meta_cols[0].markdown(f"**Source:** `{sample['source']}`")
meta_cols[1].markdown(f"**Split file:** `{sample['split_file']}`")
meta_cols[2].markdown(f"**Answer type:** `{answer_type(sample['solution'])}`")
img_basename = os.path.basename(sample["image"])
meta_cols[3].markdown(f"**Image:** `{img_basename}`")

# ---------------------------------------------------------------------------
# Question and Answer
# ---------------------------------------------------------------------------
qa_col, img_col = st.columns([1, 1])

with qa_col:
    st.markdown("#### Question")
    st.info(sample["problem"])

    st.markdown("#### Answer")
    ans = sample["solution"]
    if is_numeric_answer(ans):
        st.success(f"{ans}  (numeric)")
    else:
        st.success(f"{ans}  (text)")

with img_col:
    st.markdown("#### Chart Image")
    img_path = sample["image"]
    if os.path.isfile(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.error(f"Image not found: {img_path}")

# ---------------------------------------------------------------------------
# CSV Data Table
# ---------------------------------------------------------------------------
st.markdown("#### Data Table (CSV)")
csv_path = sample.get("csv_path", "")
if csv_path and os.path.isfile(csv_path):
    try:
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
else:
    st.warning(f"CSV not found: {csv_path}")

# ---------------------------------------------------------------------------
# Raw JSON expander
# ---------------------------------------------------------------------------
with st.expander("Raw sample JSON"):
    st.json(sample)
