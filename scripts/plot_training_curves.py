#!/usr/bin/env python
"""Extract training metrics from GRPO logs and plot Figure 3:
Row 3 (outcome-only) vs Row 4 (ChartVCR) reward curves.

Usage:
  python scripts/plot_training_curves.py
"""
import re
import json
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "logs"
ROW3_LOG = f"{LOG_DIR}/grpo_row3_v6b.log"
ROW4_LOG = f"{LOG_DIR}/grpo_cvr_v6b.log"
OUT_PNG = "results/figure3_training_curves.png"


def parse_metrics(log_path):
    """Extract per-logging-step metrics from TRL GRPO log."""
    metrics = []
    with open(log_path) as f:
        content = f.read()

    # Find all JSON-like metric dicts in log
    for match in re.finditer(r"\{['\"]loss['\"].*?\}", content, re.DOTALL):
        try:
            # TRL logs use single quotes — convert to valid JSON
            raw = match.group()
            raw = raw.replace("'", '"')
            d = json.loads(raw)
            metrics.append(d)
        except json.JSONDecodeError:
            continue
    return metrics


def main():
    import os
    os.makedirs("results", exist_ok=True)

    row3 = parse_metrics(ROW3_LOG)
    row4 = parse_metrics(ROW4_LOG)

    print(f"Row 3: {len(row3)} metric entries")
    print(f"Row 4: {len(row4)} metric entries")

    # Extract key metrics
    def get_series(metrics, reward_key_prefix="reward"):
        steps = list(range(len(metrics)))
        rewards = []
        frac_zero = []
        grad_norms = []
        losses = []
        for m in metrics:
            # Find reward mean — key varies by run
            r = None
            for k, v in m.items():
                if "reward" in k and "mean" in k and "std" not in k:
                    r = float(v)
                    break
            if r is None:
                r = float(m.get("reward", 0))
            rewards.append(r)
            frac_zero.append(float(m.get("frac_reward_zero_std", 0)))
            grad_norms.append(float(m.get("grad_norm", 0)))
            losses.append(float(m.get("loss", 0)))
        return steps, rewards, frac_zero, grad_norms, losses

    s3, r3, fz3, gn3, l3 = get_series(row3)
    s4, r4, fz4, gn4, l4 = get_series(row4)

    # Smooth with moving average
    def smooth(arr, window=5):
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window)/window, mode='valid').tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GRPO Training: Outcome-Only vs ChartVCR Process Reward", fontsize=14, fontweight='bold')

    # 1. Reward Mean
    ax = axes[0, 0]
    ax.plot(smooth(r3, 10), label="Row 3: Outcome-Only", color='#e74c3c', alpha=0.8)
    ax.plot(smooth(r4, 3), label="Row 4: ChartVCR", color='#2ecc71', alpha=0.8)
    ax.set_xlabel("Logging Step (×5 train steps)")
    ax.set_ylabel("Reward Mean")
    ax.set_title("(a) Reward Mean")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. frac_reward_zero_std — THE KEY METRIC
    ax = axes[0, 1]
    ax.plot(smooth(fz3, 10), label="Row 3: Outcome-Only", color='#e74c3c', alpha=0.8)
    ax.plot(smooth(fz4, 3), label="Row 4: ChartVCR", color='#2ecc71', alpha=0.8)
    ax.set_xlabel("Logging Step (×5 train steps)")
    ax.set_ylabel("Fraction of Batches with Zero Reward Std")
    ax.set_title("(b) Reward Signal Loss Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

    # 3. Gradient Norm
    ax = axes[1, 0]
    gn3_clip = [min(g, 5.0) for g in gn3]  # clip for visibility
    gn4_clip = [min(g, 5.0) for g in gn4]
    ax.plot(smooth(gn3_clip, 10), label="Row 3: Outcome-Only", color='#e74c3c', alpha=0.8)
    ax.plot(smooth(gn4_clip, 3), label="Row 4: ChartVCR", color='#2ecc71', alpha=0.8)
    ax.set_xlabel("Logging Step (×5 train steps)")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("(c) Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Loss
    ax = axes[1, 1]
    ax.plot(smooth(l3, 10), label="Row 3: Outcome-Only", color='#e74c3c', alpha=0.8)
    ax.plot(smooth(l4, 3), label="Row 4: ChartVCR", color='#2ecc71', alpha=0.8)
    ax.set_xlabel("Logging Step (×5 train steps)")
    ax.set_ylabel("Loss")
    ax.set_title("(d) Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")

    # Print summary stats for paper
    print("\n=== Summary Stats for Paper ===")
    print(f"\nRow 3 (Outcome-Only):")
    print(f"  Avg reward: {np.mean(r3):.4f}")
    print(f"  Avg frac_zero_std: {np.mean(fz3):.4f}")
    print(f"  Avg grad_norm: {np.mean(gn3):.6f}")
    print(f"\nRow 4 (ChartVCR):")
    print(f"  Avg reward: {np.mean(r4):.4f}")
    print(f"  Avg frac_zero_std: {np.mean(fz4):.4f}")
    print(f"  Avg grad_norm: {np.mean(gn4):.6f}")

    print(f"\nKey finding: frac_reward_zero_std ratio = {np.mean(fz3)/max(np.mean(fz4), 0.001):.1f}x")


if __name__ == "__main__":
    main()
