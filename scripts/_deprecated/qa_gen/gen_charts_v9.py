#!/usr/bin/env python
"""
Chart generation v9: Extended chart types for v9 data pipeline.

Sources:
  - kaggle_ext: Additional basic charts (bar/line/pie/donut) for Block A/B
  - scientific_ext: 8 new academic chart types for Block C (CharXiv targeting)
  - plotly_complex: 5 complex chart types via plotly for Block D (CQA-Pro/ChartMuseum)

All produce PNG + CSV + manifest JSONL.
5 visual styles: Academic, Business, Dark, Minimal, Default.

Usage:
  python scripts/gen_charts_v9.py --source kaggle_ext --limit 2000
  python scripts/gen_charts_v9.py --source scientific_ext --limit 3500
  python scripts/gen_charts_v9.py --source plotly_complex --limit 2000
  python scripts/gen_charts_v9.py --source all
  python scripts/gen_charts_v9.py --export_eligible   # export eligible JSON for generate_qa.py
"""
import argparse
import json
import os
import random
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

BASE = Path("/ex_disk2/mhpark/poc/chartvr")
OUTPUT_BASE = BASE / "data" / "charts_v9"

# ═══════════════════════════════════════════
# Visual Styles (5 named styles)
# ═══════════════════════════════════════════

STYLE_PRESETS = {
    "academic": {
        "style": "seaborn-v0_8-paper",
        "dpi": 150,
        "figsize": (10, 6),
        "fontsize": 11,
        "grid": True,
        "font_family": "serif",
        "spines_top": False,
        "spines_right": False,
    },
    "business": {
        "style": "default",
        "dpi": 120,
        "figsize": (10, 6),
        "fontsize": 12,
        "grid": False,
        "font_family": "sans-serif",
        "spines_top": False,
        "spines_right": False,
    },
    "dark": {
        "style": "dark_background",
        "dpi": 100,
        "figsize": (10, 7),
        "fontsize": 11,
        "grid": True,
        "font_family": "sans-serif",
        "spines_top": True,
        "spines_right": True,
    },
    "minimal": {
        "style": "seaborn-v0_8-whitegrid",
        "dpi": 100,
        "figsize": (9, 6),
        "fontsize": 10,
        "grid": False,
        "font_family": "sans-serif",
        "spines_top": False,
        "spines_right": False,
    },
    "default": {
        "style": "default",
        "dpi": random.choice([80, 100, 120]),
        "figsize": random.choice([(8, 5), (10, 6), (10, 7)]),
        "fontsize": random.choice([9, 10, 11, 12]),
        "grid": random.choice([True, False]),
        "font_family": "sans-serif",
        "spines_top": True,
        "spines_right": True,
    },
}

STYLE_NAMES = list(STYLE_PRESETS.keys())


def _rand_style():
    """Pick a random style preset."""
    name = random.choice(STYLE_NAMES)
    cfg = dict(STYLE_PRESETS[name])
    # Randomize figsize and dpi slightly for diversity
    cfg["figsize"] = random.choice([(8, 5), (10, 6), (10, 7), (12, 7), (9, 6)])
    cfg["dpi"] = random.choice([80, 100, 120, 150])
    cfg["fontsize"] = random.choice([9, 10, 11, 12])
    cfg["style_name"] = name
    return cfg


def _rand_academic_style():
    """Force academic style for Block C."""
    cfg = dict(STYLE_PRESETS["academic"])
    cfg["figsize"] = random.choice([(8, 6), (10, 6), (10, 7), (9, 6)])
    cfg["dpi"] = random.choice([100, 120, 150])
    cfg["fontsize"] = random.choice([10, 11, 12])
    cfg["style_name"] = "academic"
    return cfg


def _apply_style(ax, cfg):
    """Apply style config to axes."""
    if cfg.get("grid"):
        ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=cfg["fontsize"] - 2)
    ax.spines["top"].set_visible(cfg.get("spines_top", True))
    ax.spines["right"].set_visible(cfg.get("spines_right", True))


def _style_context(cfg):
    """Return matplotlib style context + rcParams."""
    return plt.style.context(cfg["style"])


# ═══════════════════════════════════════════
# Domains & Metadata
# ═══════════════════════════════════════════

SCIENTIFIC_DOMAINS = [
    "Physics", "Chemistry", "Biology", "Medicine", "Environmental Science",
    "Materials Science", "Astronomy", "Geology", "Neuroscience", "Ecology",
    "Computer Science", "Economics", "Psychology", "Sociology", "Engineering",
]

METHOD_NAMES = [
    ["Baseline", "Proposed", "Method-A", "Method-B", "Method-C"],
    ["CNN", "RNN", "Transformer", "MLP", "GNN"],
    ["Linear", "Quadratic", "Cubic", "Exponential", "Logarithmic"],
    ["Control", "Treatment-1", "Treatment-2", "Treatment-3", "Placebo"],
    ["Model-S", "Model-M", "Model-L", "Model-XL", "Ensemble"],
]

METRIC_NAMES = [
    ("Accuracy (%)", 50, 99), ("F1 Score", 0.4, 0.98),
    ("BLEU Score", 10, 50), ("Loss", 0.01, 2.0),
    ("AUC", 0.5, 0.99), ("Precision (%)", 50, 99),
    ("Recall (%)", 50, 99), ("RMSE", 0.1, 5.0),
    ("Perplexity", 5, 100), ("MSE", 0.01, 3.0),
]

KAGGLE_DOMAINS = [
    ("Sales Revenue by Product", ["Electronics", "Clothing", "Food", "Books", "Sports", "Home", "Toys", "Garden", "Auto", "Health"], "Revenue ($K)", (10, 500), 6),
    ("Employee Count by Department", ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Legal", "Support", "R&D", "Design"], "Employees", (20, 300), 7),
    ("Website Traffic by Source", ["Organic", "Direct", "Social", "Referral", "Email", "Paid Search", "Display", "Affiliate"], "Visits (K)", (5, 200), 5),
    ("Customer Satisfaction Score", ["Product A", "Product B", "Product C", "Product D", "Product E", "Product F"], "Score", (60, 99), 5),
    ("Monthly Expenses", ["Rent", "Utilities", "Salaries", "Marketing", "Insurance", "Equipment", "Travel", "Supplies"], "Amount ($K)", (2, 80), 6),
    ("Test Scores by Subject", ["Math", "Science", "English", "History", "Art", "Music", "PE", "Computer"], "Average Score", (50, 100), 6),
    ("Crime Rate by City", ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"], "Rate per 100K", (100, 800), 7),
    ("Agricultural Yield", ["Wheat", "Rice", "Corn", "Soybean", "Barley", "Cotton", "Sugarcane", "Potato"], "Tons/Hectare", (1, 15), 5),
    ("Hospital Admissions", ["Cardiology", "Orthopedics", "Neurology", "Oncology", "Pediatrics", "Emergency", "Surgery", "Internal"], "Admissions/Month", (50, 500), 6),
    ("Energy Consumption", ["Residential", "Commercial", "Industrial", "Transportation", "Agriculture"], "TWh", (10, 200), 5),
    ("Student Enrollment", ["CS", "Business", "Engineering", "Biology", "Psychology", "Economics", "Math", "Physics", "Chemistry", "English"], "Students", (100, 2000), 6),
    ("Airline Passengers", ["NYC-LAX", "NYC-CHI", "LAX-SFO", "MIA-NYC", "DFW-ATL", "ORD-DEN", "SEA-LAX", "BOS-DCA"], "Passengers (K)", (50, 800), 6),
    ("Water Usage by Region", ["North", "South", "East", "West", "Central", "Northeast", "Southeast", "Southwest"], "Million Gallons", (10, 500), 5),
    ("Patent Filings", ["Tech", "Pharma", "Auto", "Energy", "Telecom", "Finance", "Aerospace", "Materials"], "Filings", (100, 5000), 6),
    ("Tourism Revenue", ["France", "Spain", "USA", "China", "Italy", "Turkey", "Mexico", "Thailand", "Germany", "UK"], "Revenue ($B)", (10, 80), 7),
]

KAGGLE_TIMESERIES = [
    ("Monthly Temperature", "Temperature (°C)", (-5, 35), 12),
    ("Quarterly GDP Growth", "GDP Growth (%)", (-2, 8), 8),
    ("Daily Stock Price", "Price ($)", (50, 300), 20),
    ("Weekly App Downloads", "Downloads (K)", (5, 100), 15),
    ("Annual CO2 Emissions", "Emissions (Mt)", (100, 500), 10),
    ("Monthly Rainfall", "Rainfall (mm)", (10, 200), 12),
    ("Hourly Power Demand", "Demand (GW)", (20, 80), 24),
    ("Quarterly Profit Margin", "Margin (%)", (5, 25), 8),
]


# ═══════════════════════════════════════════
# Source 1: kaggle_ext — Basic charts for Block A/B
# ═══════════════════════════════════════════

def _gen_kaggle_categorical(idx):
    domain = KAGGLE_DOMAINS[idx % len(KAGGLE_DOMAINS)]
    title, cats, val_name, val_range, n = domain
    n = random.randint(max(3, n - 2), min(len(cats), n + 3))
    selected = random.sample(cats, n)
    values = [round(random.uniform(*val_range), 1) for _ in selected]
    df = pd.DataFrame({"Category": selected, val_name: values})
    chart_type = random.choice(["bar", "hbar", "pie", "donut", "stacked_bar", "grouped_bar"])
    return df, f"{title} ({random.randint(2020, 2025)})", chart_type, val_name


def _gen_kaggle_timeseries(idx):
    ts = KAGGLE_TIMESERIES[idx % len(KAGGLE_TIMESERIES)]
    title, val_name, val_range, n_points = ts
    n = random.randint(max(5, n_points - 3), n_points + 3)
    base = random.uniform(*val_range)
    values = [round(base + np.cumsum(np.random.randn(1))[0] * (val_range[1] - val_range[0]) * 0.1, 2) for _ in range(n)]
    values = [round(max(val_range[0], min(val_range[1], v)), 2) for v in values]
    periods = [f"P{i+1}" for i in range(n)]
    df = pd.DataFrame({"Period": periods, val_name: values})
    chart_type = random.choice(["line", "area"])
    return df, f"{title} Trend", chart_type, val_name


def _gen_kaggle_multivar(idx):
    domain = KAGGLE_DOMAINS[idx % len(KAGGLE_DOMAINS)]
    title, cats, _, val_range, n = domain
    n = random.randint(3, min(6, len(cats)))
    selected = random.sample(cats, n)
    var_names = random.sample(["2022", "2023", "2024", "Q1", "Q2", "Q3", "Q4", "Jan", "Jun", "Dec"], 2)
    data = {"Category": selected}
    for vn in var_names:
        data[vn] = [round(random.uniform(*val_range), 1) for _ in selected]
    df = pd.DataFrame(data)
    return df, f"{title}: {var_names[0]} vs {var_names[1]}", "grouped_bar", None


def render_kaggle_ext_chart(df, title, path, chart_type, style_cfg):
    """Render basic chart types including donut and stacked_bar."""
    with _style_context(style_cfg):
        fig, ax = plt.subplots(figsize=style_cfg["figsize"])
        _apply_style(ax, style_cfg)

        if chart_type == "bar":
            bars = ax.bar(df.iloc[:, 0].astype(str), df.iloc[:, 1])
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=style_cfg["fontsize"]-2)
            ax.set_ylabel(df.columns[1], fontsize=style_cfg["fontsize"])
            plt.xticks(rotation=30, fontsize=style_cfg["fontsize"]-2)

        elif chart_type == "hbar":
            bars = ax.barh(df.iloc[:, 0].astype(str), df.iloc[:, 1])
            for bar in bars:
                w = bar.get_width()
                ax.text(w, bar.get_y() + bar.get_height()/2, f" {w:.1f}",
                        ha="left", va="center", fontsize=style_cfg["fontsize"]-2)
            ax.set_xlabel(df.columns[1], fontsize=style_cfg["fontsize"])

        elif chart_type == "pie":
            wedges, texts, autotexts = ax.pie(
                df.iloc[:, 1].abs(), labels=df.iloc[:, 0], autopct='%1.1f%%',
                textprops={"fontsize": style_cfg["fontsize"]-2})

        elif chart_type == "donut":
            wedges, texts, autotexts = ax.pie(
                df.iloc[:, 1].abs(), labels=df.iloc[:, 0], autopct='%1.1f%%',
                wedgeprops=dict(width=0.4),
                textprops={"fontsize": style_cfg["fontsize"]-2})

        elif chart_type == "line":
            ax.plot(df.iloc[:, 0].astype(str), df.iloc[:, 1], marker="o", markersize=4)
            for x, y in zip(df.iloc[:, 0].astype(str), df.iloc[:, 1]):
                ax.annotate(f"{y}", (x, y), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=style_cfg["fontsize"]-3)
            ax.set_ylabel(df.columns[1], fontsize=style_cfg["fontsize"])
            plt.xticks(rotation=45, fontsize=style_cfg["fontsize"]-2)

        elif chart_type == "area":
            ax.fill_between(range(len(df)), df.iloc[:, 1], alpha=0.4)
            ax.plot(range(len(df)), df.iloc[:, 1], marker="o", markersize=3)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df.iloc[:, 0].astype(str), rotation=45, fontsize=style_cfg["fontsize"]-2)
            ax.set_ylabel(df.columns[1], fontsize=style_cfg["fontsize"])

        elif chart_type == "stacked_bar":
            if len(df.columns) > 2:
                x = np.arange(len(df))
                bottom = np.zeros(len(df))
                for col in df.columns[1:]:
                    ax.bar(x, df[col], bottom=bottom, label=col)
                    bottom += df[col].values
                ax.set_xticks(x)
                ax.set_xticklabels(df.iloc[:, 0].astype(str), rotation=30)
                ax.legend(fontsize=style_cfg["fontsize"]-2)
            else:
                ax.bar(df.iloc[:, 0].astype(str), df.iloc[:, 1])
                plt.xticks(rotation=30)

        elif chart_type == "grouped_bar":
            cats = df.iloc[:, 0]
            x = np.arange(len(cats))
            n_vars = len(df.columns) - 1
            if n_vars < 1:
                n_vars = 1
            width = 0.8 / n_vars
            for i, col in enumerate(df.columns[1:]):
                ax.bar(x + i * width, df[col], width, label=col)
            ax.set_xticks(x + width * n_vars / 2)
            ax.set_xticklabels(cats.astype(str), rotation=30, fontsize=style_cfg["fontsize"]-2)
            ax.legend(fontsize=style_cfg["fontsize"]-2)

        ax.set_title(title[:60], fontsize=style_cfg["fontsize"]+1)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)
    return True


def collect_kaggle_ext(limit=2000, tag="kext", seed=90):
    """Generate extended basic charts. Use different tag/seed per block for diversity."""
    out_dir = OUTPUT_BASE / f"kaggle_{tag}"
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    done = set()
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["slug"])
                except Exception:
                    pass

    collected = []
    errors = 0
    out_f = open(manifest_path, "a")
    random.seed(seed)
    np.random.seed(seed)

    generators = [
        ("cat", _gen_kaggle_categorical, 0.45),
        ("ts", _gen_kaggle_timeseries, 0.30),
        ("mv", _gen_kaggle_multivar, 0.25),
    ]

    for i in tqdm(range(limit), desc=f"Kaggle-{tag}", unit="chart"):
        slug = f"{tag}_{i:05d}"
        if slug in done:
            continue
        try:
            gen_type = random.choices(
                [g[0] for g in generators],
                weights=[g[2] for g in generators]
            )[0]
            gen_fn = next(g[1] for g in generators if g[0] == gen_type)
            result = gen_fn(i)
            df, title, chart_type = result[0], result[1], result[2]

            if len(df) < 2:
                errors += 1
                continue

            csv_path = out_dir / "tables" / f"{slug}.csv"
            df.to_csv(csv_path, index=False)

            style_cfg = _rand_style()
            img_path = out_dir / "png" / f"{slug}.png"
            ok = render_kaggle_ext_chart(df, title, img_path, chart_type, style_cfg)
            if not ok:
                errors += 1
                continue

            meta = {
                "slug": slug,
                "title": title,
                "source": f"kaggle_{tag}",
                "chart_type": chart_type,
                "gen_type": gen_type,
                "style": style_cfg.get("style_name", style_cfg["style"]),
                "csv_path": str(csv_path),
                "image_path": str(img_path),
                "n_rows": len(df),
                "n_cols": len(df.columns),
            }
            out_f.write(json.dumps(meta, default=str) + "\n")
            out_f.flush()
            collected.append(meta)

        except Exception as e:
            errors += 1

    out_f.close()
    print(f"  Kaggle-{tag}: {len(collected)} new + {len(done)} existing, errors={errors}")
    return collected


# ═══════════════════════════════════════════
# Source 2: scientific_ext — 8 new academic chart types for Block C
# ═══════════════════════════════════════════

SCIENTIFIC_EXT_TYPES = [
    "scatter_regression", "subplot_panel", "heatmap_corr",
    "errorbar_comparison", "logscale_plot", "dual_axis",
    "confidence_band", "violin_comparison",
]


def _gen_scatter_regression(idx):
    """Scatter + np.polyfit + R² annotation."""
    domain = SCIENTIFIC_DOMAINS[idx % len(SCIENTIFIC_DOMAINS)]
    n = random.randint(15, 40)
    slope = random.uniform(0.5, 5)
    intercept = random.uniform(-10, 30)
    noise = random.uniform(1, 8)

    x = np.sort(np.random.uniform(0, 20, n))
    y_true = slope * x + intercept
    y = y_true + np.random.randn(n) * noise

    coeffs = np.polyfit(x, y, 1)
    fitted = np.polyval(coeffs, x)
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0

    x_label = random.choice(["Concentration (mg/L)", "Dose (μg)", "Time (hours)", "Temperature (°C)", "Pressure (kPa)"])
    y_label = random.choice(["Response", "Absorbance", "Yield (%)", "Activity (U/mL)", "Intensity (a.u.)"])
    title = f"{domain}: {y_label.split('(')[0].strip()} vs {x_label.split('(')[0].strip()} (R²={r_squared:.3f})"

    df = pd.DataFrame({
        x_label: np.round(x, 3),
        y_label: np.round(y, 3),
        f"Fitted_{y_label}": np.round(fitted, 3),
    })
    return df, title, x_label, y_label, r_squared, coeffs


def _gen_subplot_panel(idx):
    """2x2 subplot panel with 4 related metrics."""
    domain = SCIENTIFIC_DOMAINS[idx % len(SCIENTIFIC_DOMAINS)]
    n = random.randint(8, 20)
    x = np.linspace(0, 10, n)

    panel_data = []
    panel_labels = random.sample([
        "Training Loss", "Validation Loss", "Accuracy", "Learning Rate",
        "Precision", "Recall", "F1 Score", "Throughput",
    ], 4)

    rows = []
    for pi, label in enumerate(panel_labels):
        if "Loss" in label:
            y = np.exp(-0.3 * x) * random.uniform(1, 5) + np.random.randn(n) * 0.1
        elif "Rate" in label:
            y = np.ones(n) * random.uniform(0.001, 0.01) * np.where(x > 5, 0.1, 1.0)
        else:
            y = 1 - np.exp(-0.4 * x) * random.uniform(0.3, 0.8) + np.random.randn(n) * 0.02
        y = np.round(y, 4)
        for xi, yi in zip(x, y):
            rows.append({"panel": label, "Epoch": round(xi, 2), "Value": round(float(yi), 4)})

    df = pd.DataFrame(rows)
    title = f"{domain}: Training Diagnostics"
    return df, title, panel_labels


def _gen_heatmap_corr(idx):
    """Correlation matrix heatmap."""
    domain = SCIENTIFIC_DOMAINS[idx % len(SCIENTIFIC_DOMAINS)]
    n_vars = random.randint(4, 8)
    var_names = random.sample([
        "Var_A", "Var_B", "Var_C", "Var_D", "Var_E", "Var_F",
        "Temp", "pH", "Conc", "Time", "Pressure", "Yield",
        "Growth", "Density", "Volume", "Mass",
    ], n_vars)

    # Generate a valid correlation matrix
    data = np.random.randn(50, n_vars)
    # Add some correlations
    for i in range(1, n_vars):
        if random.random() > 0.5:
            data[:, i] = data[:, 0] * random.uniform(0.3, 0.9) + data[:, i] * random.uniform(0.1, 0.7)
    corr = np.corrcoef(data.T)
    corr = np.round(corr, 2)

    df = pd.DataFrame(corr, index=var_names, columns=var_names)
    title = f"{domain}: Correlation Matrix"
    return df, title, var_names


def _gen_errorbar_comparison(idx):
    """N methods compared on M metrics with error bars."""
    domain = SCIENTIFIC_DOMAINS[idx % len(SCIENTIFIC_DOMAINS)]
    methods = random.choice(METHOD_NAMES)
    n_methods = random.randint(3, len(methods))
    methods = methods[:n_methods]

    metric_info = random.choice(METRIC_NAMES)
    metric_name, lo, hi = metric_info

    means = [round(random.uniform(lo, hi), 2) for _ in methods]
    stds = [round(random.uniform(0.5, (hi-lo)*0.1), 2) for _ in methods]

    df = pd.DataFrame({
        "Method": methods,
        metric_name: means,
        f"{metric_name}_std": stds,
    })
    title = f"{domain}: {metric_name.split('(')[0].strip()} Comparison"
    return df, title, metric_name


def _gen_logscale_plot(idx):
    """Log-scale y-axis plot."""
    domain = SCIENTIFIC_DOMAINS[idx % len(SCIENTIFIC_DOMAINS)]
    n = random.randint(10, 30)
    x = np.arange(1, n + 1)

    pattern = random.choice(["exp_decay", "power_law", "convergence"])
    if pattern == "exp_decay":
        y = random.uniform(100, 10000) * np.exp(-random.uniform(0.1, 0.5) * x)
        x_label, y_label = "Iteration", "Loss"
        title = f"{domain}: Training Loss (log scale)"
    elif pattern == "power_law":
        y = random.uniform(1, 100) * x.astype(float) ** random.uniform(1.5, 3.0)
        x_label, y_label = "Population Size", "Frequency"
        title = f"{domain}: Power Law Distribution"
    else:
        y = random.uniform(0.5, 5.0) / (x.astype(float) ** random.uniform(0.5, 1.5)) + random.uniform(0.01, 0.1)
        x_label, y_label = "Epoch", "Validation Error"
        title = f"{domain}: Convergence (log scale)"

    y = np.round(y + np.abs(np.random.randn(n)) * y * 0.05, 4)
    y = np.maximum(y, 1e-6)

    df = pd.DataFrame({x_label: x, y_label: y})
    return df, title, x_label, y_label


def _gen_dual_axis_scientific(idx):
    """Dual y-axis: bar + line with different scales."""
    domain = SCIENTIFIC_DOMAINS[idx % len(SCIENTIFIC_DOMAINS)]
    n = random.randint(6, 12)
    x_vals = list(range(1, n + 1))
    x_label = random.choice(["Epoch", "Experiment", "Trial", "Round"])

    m1_name = random.choice(["Loss", "Error Rate", "MSE", "Perplexity"])
    m2_name = random.choice(["Accuracy (%)", "F1 Score", "Precision (%)", "AUC"])

    v1 = np.sort(np.random.uniform(0.1, 3.0, n))[::-1]  # decreasing
    v2 = np.sort(np.random.uniform(50, 98, n))  # increasing
    v1, v2 = np.round(v1, 3), np.round(v2, 2)

    df = pd.DataFrame({x_label: x_vals, m1_name: v1, m2_name: v2})
    title = f"{domain}: {m1_name} and {m2_name} Over {x_label}s"
    return df, title, x_label, m1_name, m2_name


def _gen_confidence_band(idx):
    """Line + fill_between for confidence interval."""
    domain = SCIENTIFIC_DOMAINS[idx % len(SCIENTIFIC_DOMAINS)]
    n = random.randint(15, 30)
    x = np.linspace(0, 10, n)

    n_series = random.randint(1, 3)
    series_names = random.sample(["Method A", "Method B", "Method C", "Ours", "Baseline"], n_series)

    rows = []
    for sname in series_names:
        base = np.cumsum(np.random.uniform(0.5, 2, n))
        ci = np.random.uniform(0.5, 3, n)
        for xi, bi, ci_val in zip(x, base, ci):
            rows.append({
                "x": round(float(xi), 2),
                "Method": sname,
                "y": round(float(bi), 3),
                "y_lower": round(float(bi - ci_val), 3),
                "y_upper": round(float(bi + ci_val), 3),
            })

    df = pd.DataFrame(rows)
    x_label = random.choice(["Time (s)", "Epoch", "Step", "Distance (km)"])
    y_label = random.choice(["Performance", "Score", "Value", "Metric"])
    title = f"{domain}: {y_label} with Confidence Intervals"
    return df, title, x_label, y_label, series_names


def _gen_violin_comparison(idx):
    """Violin plot comparing distributions."""
    domain = SCIENTIFIC_DOMAINS[idx % len(SCIENTIFIC_DOMAINS)]
    methods = random.choice(METHOD_NAMES)
    n_methods = random.randint(3, min(5, len(methods)))
    methods = methods[:n_methods]
    n_samples = random.randint(20, 50)

    rows = []
    metric = random.choice(METRIC_NAMES)
    metric_name, lo, hi = metric

    for method in methods:
        center = random.uniform(lo, hi)
        spread = random.uniform((hi-lo)*0.05, (hi-lo)*0.2)
        values = np.random.normal(center, spread, n_samples)
        values = np.clip(values, lo, hi)
        for v in values:
            rows.append({"Method": method, metric_name: round(float(v), 3)})

    df = pd.DataFrame(rows)
    title = f"{domain}: {metric_name.split('(')[0].strip()} Distribution by Method"
    return df, title, metric_name


# Scientific ext renderers

def _render_scatter_regression(df, title, path, style_cfg, meta_extra):
    """Render scatter + regression line + R²."""
    x_label, y_label = df.columns[0], df.columns[1]
    r_sq = meta_extra.get("r_squared", 0)
    coeffs = meta_extra.get("coeffs", [1, 0])

    with _style_context(style_cfg):
        fig, ax = plt.subplots(figsize=style_cfg["figsize"])
        _apply_style(ax, style_cfg)
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], s=30, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=5)
        x_fit = np.linspace(df.iloc[:, 0].min(), df.iloc[:, 0].max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, "r--", linewidth=1.5, label=f"Fit (R²={r_sq:.3f})")
        ax.set_xlabel(x_label, fontsize=style_cfg["fontsize"])
        ax.set_ylabel(y_label, fontsize=style_cfg["fontsize"])
        ax.set_title(title[:70], fontsize=style_cfg["fontsize"]+1)
        ax.legend(fontsize=style_cfg["fontsize"]-2)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)


def _render_subplot_panel(df, title, path, style_cfg, meta_extra):
    """Render 2x2 subplot panel."""
    panel_labels = meta_extra["panel_labels"]
    with _style_context(style_cfg):
        fig, axes = plt.subplots(2, 2, figsize=(style_cfg["figsize"][0]+2, style_cfg["figsize"][1]+2))
        for ax, label in zip(axes.flat, panel_labels):
            sub = df[df["panel"] == label]
            ax.plot(sub["Epoch"], sub["Value"], marker="o", markersize=3, linewidth=1.5)
            ax.set_title(label, fontsize=style_cfg["fontsize"]-1)
            ax.tick_params(labelsize=style_cfg["fontsize"]-3)
            ax.grid(True, alpha=0.3)
        fig.suptitle(title[:70], fontsize=style_cfg["fontsize"]+1)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)


def _render_heatmap_corr(df, title, path, style_cfg, meta_extra):
    """Render correlation matrix heatmap."""
    with _style_context(style_cfg):
        fig, ax = plt.subplots(figsize=style_cfg["figsize"])
        if HAS_SEABORN:
            sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                       vmin=-1, vmax=1, ax=ax,
                       annot_kws={"size": style_cfg["fontsize"]-3})
        else:
            im = ax.imshow(df.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(df.columns)))
            ax.set_yticks(range(len(df.index)))
            ax.set_xticklabels(df.columns, rotation=45, ha="right")
            ax.set_yticklabels(df.index)
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    ax.text(j, i, f"{df.iloc[i, j]:.2f}", ha="center", va="center",
                           fontsize=style_cfg["fontsize"]-3)
            fig.colorbar(im, ax=ax)
        ax.set_title(title[:70], fontsize=style_cfg["fontsize"]+1)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)


def _render_errorbar_comparison(df, title, path, style_cfg, meta_extra):
    """Render error bar comparison chart."""
    metric_name = meta_extra["metric_name"]
    std_col = f"{metric_name}_std"

    with _style_context(style_cfg):
        fig, ax = plt.subplots(figsize=style_cfg["figsize"])
        _apply_style(ax, style_cfg)
        x = np.arange(len(df))
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
        bars = ax.bar(x, df[metric_name], yerr=df[std_col], capsize=5,
                     color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Method"], fontsize=style_cfg["fontsize"]-1)
        ax.set_ylabel(metric_name, fontsize=style_cfg["fontsize"])
        for bar, val in zip(bars, df[metric_name]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + df[std_col].max()*0.1,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=style_cfg["fontsize"]-2)
        ax.set_title(title[:70], fontsize=style_cfg["fontsize"]+1)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)


def _render_logscale(df, title, path, style_cfg, meta_extra):
    """Render log-scale plot."""
    x_label, y_label = df.columns[0], df.columns[1]
    with _style_context(style_cfg):
        fig, ax = plt.subplots(figsize=style_cfg["figsize"])
        _apply_style(ax, style_cfg)
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], marker="o", markersize=3, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel(x_label, fontsize=style_cfg["fontsize"])
        ax.set_ylabel(y_label, fontsize=style_cfg["fontsize"])
        ax.set_title(title[:70], fontsize=style_cfg["fontsize"]+1)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)


def _render_dual_axis_sci(df, title, path, style_cfg, meta_extra):
    """Render dual y-axis chart."""
    x_label = meta_extra["x_label"]
    m1_name = meta_extra["m1_name"]
    m2_name = meta_extra["m2_name"]

    with _style_context(style_cfg):
        fig, ax1 = plt.subplots(figsize=style_cfg["figsize"])
        _apply_style(ax1, style_cfg)
        color1, color2 = "#1f77b4", "#ff7f0e"
        ax1.bar(df[x_label].astype(str), df[m1_name], color=color1, alpha=0.7, label=m1_name)
        ax1.set_ylabel(m1_name, color=color1, fontsize=style_cfg["fontsize"])
        ax1.tick_params(axis="y", labelcolor=color1)
        ax2 = ax1.twinx()
        ax2.plot(df[x_label].astype(str), df[m2_name], color=color2, marker="o", linewidth=2, label=m2_name)
        ax2.set_ylabel(m2_name, color=color2, fontsize=style_cfg["fontsize"])
        ax2.tick_params(axis="y", labelcolor=color2)
        ax1.set_xlabel(x_label, fontsize=style_cfg["fontsize"])
        ax1.set_title(title[:70], fontsize=style_cfg["fontsize"]+1)
        plt.xticks(rotation=45, fontsize=style_cfg["fontsize"]-2)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)


def _render_confidence_band(df, title, path, style_cfg, meta_extra):
    """Render confidence band plot."""
    series_names = meta_extra["series_names"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(series_names)))

    with _style_context(style_cfg):
        fig, ax = plt.subplots(figsize=style_cfg["figsize"])
        _apply_style(ax, style_cfg)
        for sname, color in zip(series_names, colors):
            sub = df[df["Method"] == sname].sort_values("x")
            ax.plot(sub["x"], sub["y"], linewidth=1.5, label=sname, color=color)
            ax.fill_between(sub["x"], sub["y_lower"], sub["y_upper"], alpha=0.2, color=color)

        ax.set_xlabel(meta_extra.get("x_label", "x"), fontsize=style_cfg["fontsize"])
        ax.set_ylabel(meta_extra.get("y_label", "y"), fontsize=style_cfg["fontsize"])
        ax.set_title(title[:70], fontsize=style_cfg["fontsize"]+1)
        ax.legend(fontsize=style_cfg["fontsize"]-2)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)


def _render_violin(df, title, path, style_cfg, meta_extra):
    """Render violin comparison plot."""
    metric_name = meta_extra["metric_name"]
    with _style_context(style_cfg):
        fig, ax = plt.subplots(figsize=style_cfg["figsize"])
        _apply_style(ax, style_cfg)
        if HAS_SEABORN:
            sns.violinplot(data=df, x="Method", y=metric_name, ax=ax, inner="box")
        else:
            methods = df["Method"].unique()
            data_groups = [df[df["Method"] == m][metric_name].values for m in methods]
            parts = ax.violinplot(data_groups, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(methods)+1))
            ax.set_xticklabels(methods)
        ax.set_ylabel(metric_name, fontsize=style_cfg["fontsize"])
        ax.set_title(title[:70], fontsize=style_cfg["fontsize"]+1)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)


def collect_scientific_ext(limit=3500):
    """Generate 8 academic chart types for Block C."""
    out_dir = OUTPUT_BASE / "scientific_ext"
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    done = set()
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["slug"])
                except Exception:
                    pass

    collected = []
    errors = 0
    out_f = open(manifest_path, "a")
    random.seed(91)
    np.random.seed(91)

    n_types = len(SCIENTIFIC_EXT_TYPES)

    for i in tqdm(range(limit), desc="Scientific-ext", unit="chart"):
        slug = f"sci_ext_{i:05d}"
        if slug in done:
            continue
        try:
            chart_type = SCIENTIFIC_EXT_TYPES[i % n_types]
            style_cfg = _rand_academic_style()
            meta_extra = {}

            if chart_type == "scatter_regression":
                df, title, x_label, y_label, r_sq, coeffs = _gen_scatter_regression(i)
                meta_extra = {"r_squared": r_sq, "coeffs": coeffs.tolist()}
                _render_scatter_regression(df, title, out_dir / "png" / f"{slug}.png", style_cfg, meta_extra)

            elif chart_type == "subplot_panel":
                df, title, panel_labels = _gen_subplot_panel(i)
                meta_extra = {"panel_labels": panel_labels}
                _render_subplot_panel(df, title, out_dir / "png" / f"{slug}.png", style_cfg, meta_extra)

            elif chart_type == "heatmap_corr":
                df, title, var_names = _gen_heatmap_corr(i)
                meta_extra = {"var_names": var_names}
                _render_heatmap_corr(df, title, out_dir / "png" / f"{slug}.png", style_cfg, meta_extra)

            elif chart_type == "errorbar_comparison":
                df, title, metric_name = _gen_errorbar_comparison(i)
                meta_extra = {"metric_name": metric_name}
                _render_errorbar_comparison(df, title, out_dir / "png" / f"{slug}.png", style_cfg, meta_extra)

            elif chart_type == "logscale_plot":
                df, title, x_label, y_label = _gen_logscale_plot(i)
                meta_extra = {}
                _render_logscale(df, title, out_dir / "png" / f"{slug}.png", style_cfg, meta_extra)

            elif chart_type == "dual_axis":
                df, title, x_label, m1_name, m2_name = _gen_dual_axis_scientific(i)
                meta_extra = {"x_label": x_label, "m1_name": m1_name, "m2_name": m2_name}
                _render_dual_axis_sci(df, title, out_dir / "png" / f"{slug}.png", style_cfg, meta_extra)

            elif chart_type == "confidence_band":
                df, title, x_label, y_label, series_names = _gen_confidence_band(i)
                meta_extra = {"x_label": x_label, "y_label": y_label, "series_names": series_names}
                _render_confidence_band(df, title, out_dir / "png" / f"{slug}.png", style_cfg, meta_extra)

            elif chart_type == "violin_comparison":
                df, title, metric_name = _gen_violin_comparison(i)
                meta_extra = {"metric_name": metric_name}
                _render_violin(df, title, out_dir / "png" / f"{slug}.png", style_cfg, meta_extra)

            else:
                errors += 1
                continue

            csv_path = out_dir / "tables" / f"{slug}.csv"
            df.to_csv(csv_path, index=False)

            meta = {
                "slug": slug,
                "title": title,
                "source": "scientific_ext",
                "chart_type": chart_type,
                "style": style_cfg.get("style_name", "academic"),
                "csv_path": str(csv_path),
                "image_path": str(out_dir / "png" / f"{slug}.png"),
                "n_rows": len(df),
                "n_cols": len(df.columns),
            }
            # Store extra metadata for QA generation
            for k, v in meta_extra.items():
                if isinstance(v, (str, int, float, bool, list)):
                    meta[k] = v

            out_f.write(json.dumps(meta, default=str) + "\n")
            out_f.flush()
            collected.append(meta)

        except Exception as e:
            errors += 1

    out_f.close()
    print(f"  Scientific-ext: {len(collected)} new + {len(done)} existing, errors={errors}")
    return collected


# ═══════════════════════════════════════════
# Source 3: plotly_complex — 5 complex chart types for Block D
# ═══════════════════════════════════════════

PLOTLY_TYPES = ["waterfall", "sankey", "sunburst", "bubble", "gauge"]


def _gen_waterfall(idx):
    """Waterfall chart data."""
    categories = ["Revenue", "COGS", "Gross Profit", "Operating Exp", "Tax",
                  "Other Income", "Net Profit"]
    n = random.randint(5, len(categories))
    cats = categories[:n]
    values = []
    measures = []
    running = 0
    for i, c in enumerate(cats):
        if i == 0:
            v = random.randint(100, 500)
            measures.append("absolute")
        elif i == n - 1:
            v = running
            measures.append("total")
        elif "Profit" in c and i > 0:
            v = running
            measures.append("total")
        else:
            v = random.randint(-200, -10) if "Exp" in c or "COGS" in c or "Tax" in c else random.randint(5, 100)
            measures.append("relative")
        running += v if measures[-1] == "relative" else (v - running if measures[-1] == "total" else v)
        values.append(v)

    df = pd.DataFrame({"Category": cats, "Value": values, "Measure": measures})
    title = f"Financial Waterfall ({random.randint(2022, 2025)})"
    return df, title


def _gen_sankey(idx):
    """Sankey diagram data."""
    sources_labels = ["Energy", "Agriculture", "Industry", "Transport", "Residential"]
    targets_labels = ["CO2", "Methane", "N2O", "Heat Loss", "Waste"]
    n_links = random.randint(6, 15)

    rows = []
    for _ in range(n_links):
        s = random.choice(sources_labels)
        t = random.choice(targets_labels)
        v = random.randint(10, 200)
        rows.append({"Source": s, "Target": t, "Value": v})

    df = pd.DataFrame(rows)
    title = "Energy Flow Diagram"
    return df, title


def _gen_sunburst(idx):
    """Sunburst chart data."""
    root = random.choice(["Company", "Organization", "Portfolio", "Budget"])
    level1 = random.sample(["Division A", "Division B", "Division C", "Division D", "Division E"], random.randint(3, 5))
    rows = [{"Label": root, "Parent": "", "Value": 0}]
    for div in level1:
        div_val = random.randint(100, 500)
        rows.append({"Label": div, "Parent": root, "Value": div_val})
        n_sub = random.randint(2, 4)
        for j in range(n_sub):
            sub_name = f"{div}-{chr(65+j)}"
            rows.append({"Label": sub_name, "Parent": div, "Value": random.randint(10, div_val // n_sub)})

    df = pd.DataFrame(rows)
    title = f"{root} Hierarchy Breakdown"
    return df, title


def _gen_bubble(idx):
    """Bubble chart data."""
    entities = random.sample([
        "USA", "China", "India", "Germany", "Japan", "Brazil", "UK",
        "France", "Canada", "Australia", "Korea", "Mexico", "Russia",
    ], random.randint(6, 10))
    x_vals = [round(random.uniform(1000, 80000), 0) for _ in entities]
    y_vals = [round(random.uniform(50, 85), 1) for _ in entities]
    sizes = [round(random.uniform(5, 1400), 1) for _ in entities]

    df = pd.DataFrame({
        "Country": entities,
        "GDP per Capita ($)": x_vals,
        "Life Expectancy": y_vals,
        "Population (M)": sizes,
    })
    title = "GDP vs Life Expectancy (bubble = population)"
    return df, title


def _gen_gauge(idx):
    """Gauge chart data."""
    metrics = [
        ("Customer Satisfaction", 0, 100, "%"),
        ("System Uptime", 90, 100, "%"),
        ("Revenue Target", 0, 150, "%"),
        ("Quality Score", 0, 10, ""),
        ("Performance Index", 0, 100, "pts"),
    ]
    m = metrics[idx % len(metrics)]
    name, lo, hi, unit = m
    value = round(random.uniform(lo + (hi-lo)*0.3, hi * 0.95), 1)
    df = pd.DataFrame({"Metric": [name], "Value": [value], "Min": [lo], "Max": [hi], "Unit": [unit]})
    title = f"{name}: {value}{unit}"
    return df, title


def _render_plotly_chart(df, title, path, chart_type):
    """Render plotly chart to PNG."""
    if not HAS_PLOTLY:
        return False

    if chart_type == "waterfall":
        fig = go.Figure(go.Waterfall(
            x=df["Category"].tolist(),
            y=df["Value"].tolist(),
            measure=df["Measure"].tolist(),
            textposition="outside",
            text=[str(v) for v in df["Value"]],
        ))
        fig.update_layout(title=title, showlegend=False)

    elif chart_type == "sankey":
        all_labels = list(set(df["Source"].tolist() + df["Target"].tolist()))
        source_idx = [all_labels.index(s) for s in df["Source"]]
        target_idx = [all_labels.index(t) for t in df["Target"]]
        fig = go.Figure(go.Sankey(
            node=dict(label=all_labels, pad=15, thickness=20),
            link=dict(source=source_idx, target=target_idx, value=df["Value"].tolist()),
        ))
        fig.update_layout(title=title)

    elif chart_type == "sunburst":
        fig = go.Figure(go.Sunburst(
            labels=df["Label"].tolist(),
            parents=df["Parent"].tolist(),
            values=df["Value"].tolist(),
        ))
        fig.update_layout(title=title)

    elif chart_type == "bubble":
        fig = go.Figure(go.Scatter(
            x=df.iloc[:, 1],
            y=df.iloc[:, 2],
            mode="markers+text",
            marker=dict(size=np.sqrt(df.iloc[:, 3]) * 2, sizemode="area", sizeref=0.5),
            text=df.iloc[:, 0],
            textposition="top center",
        ))
        fig.update_layout(title=title, xaxis_title=df.columns[1], yaxis_title=df.columns[2])

    elif chart_type == "gauge":
        row = df.iloc[0]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=row["Value"],
            title={"text": row["Metric"]},
            gauge={
                "axis": {"range": [row["Min"], row["Max"]]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [row["Min"], row["Min"] + (row["Max"]-row["Min"])*0.5], "color": "lightgray"},
                    {"range": [row["Min"] + (row["Max"]-row["Min"])*0.5, row["Max"]], "color": "gray"},
                ],
            },
        ))
        fig.update_layout(title=title)

    else:
        return False

    fig.update_layout(width=1000, height=600)
    try:
        pio.write_image(fig, str(path), format="png", scale=2)
        return True
    except Exception:
        return False


def collect_plotly_complex(limit=2000):
    """Generate complex charts via plotly for Block D."""
    if not HAS_PLOTLY:
        print("  WARNING: plotly not installed, skipping plotly_complex")
        return []

    out_dir = OUTPUT_BASE / "plotly_complex"
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    done = set()
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["slug"])
                except Exception:
                    pass

    collected = []
    errors = 0
    out_f = open(manifest_path, "a")
    random.seed(92)
    np.random.seed(92)

    n_types = len(PLOTLY_TYPES)

    for i in tqdm(range(limit), desc="Plotly-complex", unit="chart"):
        slug = f"plotly_{i:05d}"
        if slug in done:
            continue
        try:
            chart_type = PLOTLY_TYPES[i % n_types]

            if chart_type == "waterfall":
                df, title = _gen_waterfall(i)
            elif chart_type == "sankey":
                df, title = _gen_sankey(i)
            elif chart_type == "sunburst":
                df, title = _gen_sunburst(i)
            elif chart_type == "bubble":
                df, title = _gen_bubble(i)
            elif chart_type == "gauge":
                df, title = _gen_gauge(i)
            else:
                errors += 1
                continue

            csv_path = out_dir / "tables" / f"{slug}.csv"
            df.to_csv(csv_path, index=False)

            img_path = out_dir / "png" / f"{slug}.png"
            ok = _render_plotly_chart(df, title, img_path, chart_type)
            if not ok:
                errors += 1
                continue

            meta = {
                "slug": slug,
                "title": title,
                "source": "plotly_complex",
                "chart_type": chart_type,
                "csv_path": str(csv_path),
                "image_path": str(img_path),
                "n_rows": len(df),
                "n_cols": len(df.columns),
            }
            out_f.write(json.dumps(meta, default=str) + "\n")
            out_f.flush()
            collected.append(meta)

        except Exception:
            errors += 1

    out_f.close()
    print(f"  Plotly-complex: {len(collected)} new + {len(done)} existing, errors={errors}")
    return collected


# ═══════════════════════════════════════════
# Export eligible JSON for generate_qa.py
# ═══════════════════════════════════════════

def export_eligible():
    """Export eligible chart lists for API-based QA generation."""
    for source, name in [
        ("scientific_ext", "block_c_eligible.json"),
        ("plotly_complex", "block_d_eligible.json"),
    ]:
        manifest_path = OUTPUT_BASE / source / "manifest.jsonl"
        if not manifest_path.exists():
            print(f"  {manifest_path} not found, skipping")
            continue

        charts = []
        with open(manifest_path) as f:
            for line in f:
                try:
                    m = json.loads(line)
                    charts.append({
                        "slug": m["slug"],
                        "source": m["source"],
                        "csv_path": m["csv_path"],
                        "image_path": m["image_path"],
                    })
                except Exception:
                    pass

        out_path = OUTPUT_BASE / name
        with open(out_path, "w") as f:
            json.dump(charts, f, indent=2, default=str)
        print(f"  Exported {len(charts)} charts to {out_path}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Chart generation v9")
    parser.add_argument("--source", choices=[
        "kaggle_ext", "scientific_ext", "plotly_complex", "all",
        "block_a", "block_b", "block_f",
    ], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Override per-source limit")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--export_eligible", action="store_true", help="Export eligible JSON for generate_qa.py")
    args = parser.parse_args()

    if args.export_eligible:
        export_eligible()
        return

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    total = []

    # Per-block dedicated chart generation (no overlap between blocks)
    if args.source == "block_a":
        total.extend(collect_kaggle_ext(args.limit or 2000, tag="blk_a", seed=args.seed or 100))
    elif args.source == "block_b":
        total.extend(collect_kaggle_ext(args.limit or 2000, tag="blk_b", seed=args.seed or 200))
    elif args.source == "block_f":
        total.extend(collect_kaggle_ext(args.limit or 1000, tag="blk_f", seed=args.seed or 300))

    # Original sources (backward compatible)
    elif args.source in ("kaggle_ext", "all"):
        total.extend(collect_kaggle_ext(args.limit or 2000))
        if args.source == "all":
            total.extend(collect_scientific_ext(args.limit or 3500))
            total.extend(collect_plotly_complex(args.limit or 2000))
    elif args.source == "scientific_ext":
        total.extend(collect_scientific_ext(args.limit or 3500))
    elif args.source == "plotly_complex":
        total.extend(collect_plotly_complex(args.limit or 2000))

    print(f"\nTOTAL v9: {len(total)} charts generated")

    # Also export eligible lists
    export_eligible()


if __name__ == "__main__":
    main()
