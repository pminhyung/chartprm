"""
Eval Inspection Dashboard — sample-by-sample review of eval/inference results.

Launch:
    streamlit run /ex_disk2/mhpark/poc/chartvr/scripts/eval_inspection_dashboard.py --server.port 8502

Data source:
    /ex_disk2/mhpark/poc/chartvr/results/main_table/qwen25vl_zeroshot_chartqa_human.json
"""

import json
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "/ex_disk2/mhpark/poc/chartvr/results/main_table/qwen25vl_zeroshot_chartqa_human.json"


@st.cache_data
def load_data(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(
        page_title="Eval Inspection Dashboard",
        layout="wide",
    )
    st.title("Eval Inspection Dashboard")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = load_data(DATA_PATH)
    model_name: str = data.get("model", "unknown")
    benchmark: str = data.get("benchmark", "unknown")
    overall_accuracy: float = data.get("accuracy", 0.0)
    n_samples: int = data.get("n_samples", 0)
    all_results: list[dict] = data.get("results", [])

    # ------------------------------------------------------------------
    # Sidebar — model info + filters
    # ------------------------------------------------------------------
    st.sidebar.header("Model / Benchmark Info")
    st.sidebar.markdown(f"**Model**: `{model_name}`")
    st.sidebar.markdown(f"**Benchmark**: `{benchmark}`")
    st.sidebar.markdown(f"**Overall Accuracy**: `{overall_accuracy:.1%}`")
    st.sidebar.markdown(f"**Total Samples**: `{n_samples}`")
    st.sidebar.divider()

    # Correctness filter
    st.sidebar.header("Filters")
    correctness_filter = st.sidebar.radio(
        "Show predictions:",
        options=["All", "Correct only", "Incorrect only"],
        index=0,
    )

    # Search by question text
    search_query = st.sidebar.text_input("Search question text:", value="")

    # ------------------------------------------------------------------
    # Apply filters
    # ------------------------------------------------------------------
    filtered = all_results

    if correctness_filter == "Correct only":
        filtered = [r for r in filtered if r.get("accuracy", 0.0) == 1.0]
    elif correctness_filter == "Incorrect only":
        filtered = [r for r in filtered if r.get("accuracy", 0.0) != 1.0]

    if search_query.strip():
        query_lower = search_query.strip().lower()
        filtered = [r for r in filtered if query_lower in r.get("question", "").lower()]

    total_filtered = len(filtered)

    if total_filtered == 0:
        st.warning("No samples match the current filters.")
        return

    # Stats after filtering
    n_correct = sum(1 for r in filtered if r.get("accuracy", 0.0) == 1.0)
    n_incorrect = total_filtered - n_correct

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Filtered Samples", total_filtered)
    col_b.metric("Correct", n_correct)
    col_c.metric("Incorrect", n_incorrect)

    st.divider()

    # ------------------------------------------------------------------
    # Sample navigator
    # ------------------------------------------------------------------
    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 0

    # Clamp index to valid range after filter changes
    if st.session_state.sample_idx >= total_filtered:
        st.session_state.sample_idx = 0

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 2])

    with nav_col1:
        if st.button("Prev", use_container_width=True):
            st.session_state.sample_idx = max(0, st.session_state.sample_idx - 1)

    with nav_col2:
        if st.button("Next", use_container_width=True):
            st.session_state.sample_idx = min(total_filtered - 1, st.session_state.sample_idx + 1)

    with nav_col3:
        jump_idx = st.number_input(
            "Go to index:",
            min_value=0,
            max_value=total_filtered - 1,
            value=st.session_state.sample_idx,
            step=1,
            key="jump_input",
        )
        if jump_idx != st.session_state.sample_idx:
            st.session_state.sample_idx = jump_idx

    with nav_col4:
        st.markdown(f"**Showing sample {st.session_state.sample_idx + 1} / {total_filtered}**")

    idx = st.session_state.sample_idx
    sample = filtered[idx]

    st.divider()

    # ------------------------------------------------------------------
    # Sample display
    # ------------------------------------------------------------------
    is_correct = sample.get("accuracy", 0.0) == 1.0
    badge_color = "green" if is_correct else "red"
    badge_text = "CORRECT" if is_correct else "INCORRECT"

    st.markdown(
        f"### Sample #{idx + 1} &nbsp; "
        f"<span style='background-color:{badge_color};color:white;padding:2px 10px;"
        f"border-radius:4px;font-size:0.85em;'>{badge_text}</span>",
        unsafe_allow_html=True,
    )

    # Question
    st.markdown("**Question:**")
    st.info(sample.get("question", "N/A"))

    # Gold vs Predicted side by side
    ans_col1, ans_col2 = st.columns(2)

    with ans_col1:
        st.markdown("**Gold Answer:**")
        st.success(sample.get("gold_answer", "N/A"))

    with ans_col2:
        st.markdown("**Predicted Answer:**")
        if is_correct:
            st.success(sample.get("predicted_answer", "N/A"))
        else:
            st.error(sample.get("predicted_answer", "N/A"))

    # Full model response
    with st.expander("Full Model Response", expanded=False):
        st.text(sample.get("response", "N/A"))

    # ------------------------------------------------------------------
    # Quick-jump table (collapsible)
    # ------------------------------------------------------------------
    with st.expander("Browse All Filtered Samples (table view)", expanded=False):
        import pandas as pd

        table_rows = []
        for i, r in enumerate(filtered):
            table_rows.append(
                {
                    "Index": i,
                    "Question": r.get("question", "")[:100],
                    "Gold": r.get("gold_answer", ""),
                    "Predicted": r.get("predicted_answer", ""),
                    "Correct": "Yes" if r.get("accuracy", 0.0) == 1.0 else "No",
                }
            )
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
