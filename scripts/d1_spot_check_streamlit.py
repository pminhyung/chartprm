"""D1 spot check — Streamlit UI replacement for CLI.

Run:
  /ex_disk2/mhpark/poc/vllm_nightly_env/bin/streamlit run scripts/d1_spot_check_streamlit.py
  → open http://localhost:8501

Reads:  data/d1_pilot/segmented_v3.jsonl
Writes: data/d1_pilot/spot_check.json   (id -> {segments:[{idx,ok}], note, n_bad})
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import streamlit as st
from PIL import Image

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
IN_JSONL = BASE / "data/d1_pilot/segmented_v3.jsonl"
OUT_JSON = BASE / "data/d1_pilot/spot_check.json"

SEED = 0
N_SPOT = 5


@st.cache_data
def load_samples():
    data = [json.loads(l) for l in open(IN_JSONL) if l.strip()]
    data = [d for d in data if d.get("n_steps", 0) > 0]
    rng = random.Random(SEED)
    return rng.sample(data, N_SPOT)


def load_state() -> dict:
    if not OUT_JSON.exists():
        return {}
    return json.loads(OUT_JSON.read_text())


def save_state(state: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def main():
    st.set_page_config(page_title="D1 Spot Check", layout="wide")
    samples = load_samples()
    state = load_state()

    st.sidebar.title("D1 Spot Check")
    st.sidebar.markdown(f"5 random samples (seed={SEED}) from `segmented_v3.jsonl`")
    st.sidebar.markdown(f"**Progress:** {len(state)}/{N_SPOT} samples saved")

    sample_labels = [
        f"{i+1}. {s['id']}{' ✓' if s['id'] in state else ''}"
        for i, s in enumerate(samples)
    ]
    sel_idx = st.sidebar.radio("Sample", options=list(range(N_SPOT)),
                                format_func=lambda i: sample_labels[i])

    # Final summary panel
    if len(state) == N_SPOT:
        n_bad = sum(1 for r in state.values() if r["n_bad"] > 0)
        verdict = "PASS" if n_bad == 0 else ("CONDITIONAL" if n_bad == 1 else "ESCALATE")
        st.sidebar.divider()
        st.sidebar.subheader("Final verdict")
        st.sidebar.metric("BAD samples", f"{n_bad}/{N_SPOT}")
        st.sidebar.success(f"**{verdict}**" if verdict == "PASS" else f"**{verdict}**")

    s = samples[sel_idx]
    st.title(f"Sample {sel_idx+1}/{N_SPOT}")
    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown(f"**ID:** `{s['id']}`")
        st.markdown(f"**Source:** {s['source']}  /  **n_steps:** {s['n_steps']}")
        st.markdown(f"**Question:** {s['question']}")
        with st.expander("Gold answer", expanded=False):
            st.text(s["gold_answer"][:1500] + ("..." if len(s["gold_answer"]) > 1500 else ""))
    with cols[1]:
        try:
            st.image(Image.open(s["image_path"]), use_container_width=True)
        except Exception as e:
            st.error(f"Image load failed: {e}")

    st.divider()
    st.subheader("Segments — mark each as OK or BAD")

    prev = state.get(s["id"], {})
    prev_segs = {seg["idx"]: seg["ok"] for seg in prev.get("segments", [])}

    seg_choices = {}
    for i, step in enumerate(s["steps"]):
        with st.container():
            sub = st.columns([4, 1])
            with sub[0]:
                st.markdown(f"**Segment {i+1}/{s['n_steps']}** ({s['step_token_lens'][i]} tok)")
                st.text_area("", value=step, height=160, key=f"text_{s['id']}_{i}", disabled=True,
                             label_visibility="collapsed")
            with sub[1]:
                default_index = 0 if prev_segs.get(i, True) else 1
                choice = st.radio(
                    "verdict", ["OK", "BAD"], horizontal=False,
                    key=f"v_{s['id']}_{i}", index=default_index,
                    label_visibility="collapsed",
                )
                seg_choices[i] = choice == "OK"

    note = st.text_area("Note (optional)", value=prev.get("note", ""), key=f"note_{s['id']}",
                        height=70)

    save_cols = st.columns([1, 4])
    with save_cols[0]:
        if st.button("💾 Save sample", type="primary", key=f"save_{s['id']}", use_container_width=True):
            rec = {
                "id": s["id"],
                "source": s["source"],
                "n_steps": s["n_steps"],
                "segments": [{"idx": i, "ok": ok} for i, ok in seg_choices.items()],
                "n_bad": sum(1 for ok in seg_choices.values() if not ok),
                "note": note,
            }
            state = load_state()
            state[s["id"]] = rec
            save_state(state)
            st.success(f"Saved → {OUT_JSON}")
            st.rerun()
    with save_cols[1]:
        if s["id"] in state:
            r = state[s["id"]]
            st.info(f"Previously saved: n_bad={r['n_bad']}/{s['n_steps']}, note={r['note'][:60] or '—'}")


if __name__ == "__main__":
    main()
