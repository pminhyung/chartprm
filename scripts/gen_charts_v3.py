#!/usr/bin/env python
"""
Chart generation v3: New sources for v8 data expansion.

Sources:
  - kaggle_like: Diverse tabular data (economics, sports, demographics, etc.)
  - scientific: Academic-style charts (distributions, regression, time series)
  - additional: Complex charts (subplots, dual-axis, text-answer-friendly)

All sources produce PNG + CSV + manifest JSONL.
5 visual styles randomized: default, seaborn-v0_8-paper, whitegrid, bmh, dark_background.

Usage:
  python scripts/gen_charts_v3.py --source kaggle_like --limit 3000
  python scripts/gen_charts_v3.py --source scientific --limit 3000
  python scripts/gen_charts_v3.py --source additional --limit 2000
  python scripts/gen_charts_v3.py --source all
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

BASE = Path("/ex_disk2/mhpark/poc/chartvr")
OUTPUT_BASE = BASE / "data" / "charts_v2"

VISUAL_STYLES = ["default", "seaborn-v0_8-paper", "seaborn-v0_8-whitegrid", "bmh", "dark_background"]
RANDOM_DPI = [80, 100, 120, 150]
RANDOM_FIGSIZE = [(8, 5), (10, 6), (10, 7), (12, 7), (9, 6)]
RANDOM_FONTSIZE = [9, 10, 11, 12]


def _rand_style():
    return {
        "style": random.choice(VISUAL_STYLES),
        "dpi": random.choice(RANDOM_DPI),
        "figsize": random.choice(RANDOM_FIGSIZE),
        "fontsize": random.choice(RANDOM_FONTSIZE),
        "grid": random.choice([True, False]),
    }


def _apply_style(ax, cfg):
    if cfg["grid"]:
        ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=cfg["fontsize"] - 2)


# ═══════════════════════════════════════════
# Kaggle-like: Diverse tabular data
# ═══════════════════════════════════════════

KAGGLE_DOMAINS = [
    # (domain, category_names, value_name, value_range, n_categories)
    ("Sales Revenue by Product", ["Electronics", "Clothing", "Food", "Books", "Sports", "Home", "Toys", "Garden", "Auto", "Health"], "Revenue ($K)", (10, 500), 6),
    ("Employee Count by Department", ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Legal", "Support", "R&D", "Design"], "Employees", (20, 300), 7),
    ("Website Traffic by Source", ["Organic", "Direct", "Social", "Referral", "Email", "Paid Search", "Display", "Affiliate"], "Visits (K)", (5, 200), 5),
    ("Customer Satisfaction Score", ["Product A", "Product B", "Product C", "Product D", "Product E", "Product F"], "Score", (60, 99), 5),
    ("Monthly Expenses", ["Rent", "Utilities", "Salaries", "Marketing", "Insurance", "Equipment", "Travel", "Supplies"], "Amount ($K)", (2, 80), 6),
    ("Test Scores by Subject", ["Math", "Science", "English", "History", "Art", "Music", "PE", "Computer"], "Average Score", (50, 100), 6),
    ("Crime Rate by City", ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"], "Rate per 100K", (100, 800), 7),
    ("Agricultural Yield", ["Wheat", "Rice", "Corn", "Soybean", "Barley", "Cotton", "Sugarcane", "Potato"], "Tons/Hectare", (1, 15), 5),
    ("Hospital Admissions", ["Cardiology", "Orthopedics", "Neurology", "Oncology", "Pediatrics", "Emergency", "Surgery", "Internal"], "Admissions/Month", (50, 500), 6),
    ("Energy Consumption by Sector", ["Residential", "Commercial", "Industrial", "Transportation", "Agriculture"], "TWh", (10, 200), 5),
    ("Student Enrollment by Major", ["CS", "Business", "Engineering", "Biology", "Psychology", "Economics", "Math", "Physics", "Chemistry", "English"], "Students", (100, 2000), 6),
    ("Airline Passengers by Route", ["NYC-LAX", "NYC-CHI", "LAX-SFO", "MIA-NYC", "DFW-ATL", "ORD-DEN", "SEA-LAX", "BOS-DCA"], "Passengers (K)", (50, 800), 6),
    ("Water Usage by Region", ["North", "South", "East", "West", "Central", "Northeast", "Southeast", "Southwest"], "Million Gallons", (10, 500), 5),
    ("Patent Filings by Industry", ["Tech", "Pharma", "Auto", "Energy", "Telecom", "Finance", "Aerospace", "Materials"], "Filings", (100, 5000), 6),
    ("Tourism Revenue by Country", ["France", "Spain", "USA", "China", "Italy", "Turkey", "Mexico", "Thailand", "Germany", "UK"], "Revenue ($B)", (10, 80), 7),
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


def _gen_kaggle_categorical(idx):
    """Generate a categorical chart (bar/pie/horizontal bar)."""
    domain = KAGGLE_DOMAINS[idx % len(KAGGLE_DOMAINS)]
    title, cats, val_name, val_range, n = domain
    n = random.randint(max(3, n - 2), min(len(cats), n + 2))
    selected = random.sample(cats, n)
    values = [round(random.uniform(*val_range), 1) for _ in selected]
    df = pd.DataFrame({"Category": selected, val_name: values})
    chart_type = random.choice(["bar", "hbar", "pie"])
    return df, f"{title} ({random.randint(2020, 2025)})", chart_type


def _gen_kaggle_timeseries(idx):
    """Generate a time series chart (line/area)."""
    ts = KAGGLE_TIMESERIES[idx % len(KAGGLE_TIMESERIES)]
    title, val_name, val_range, n_points = ts
    n = random.randint(max(5, n_points - 3), n_points + 3)
    base = random.uniform(*val_range)
    values = [round(base + np.cumsum(np.random.randn(1))[0] * (val_range[1] - val_range[0]) * 0.1, 2) for _ in range(n)]
    # Clip to range
    values = [round(max(val_range[0], min(val_range[1], v)), 2) for v in values]
    periods = [f"P{i+1}" for i in range(n)]
    df = pd.DataFrame({"Period": periods, val_name: values})
    chart_type = random.choice(["line", "area"])
    return df, f"{title} Trend", chart_type


def _gen_kaggle_multivar(idx):
    """Generate multi-variable comparison chart."""
    domain = KAGGLE_DOMAINS[idx % len(KAGGLE_DOMAINS)]
    title, cats, _, val_range, n = domain
    n = random.randint(3, min(6, len(cats)))
    selected = random.sample(cats, n)
    var_names = random.sample(["2022", "2023", "2024", "Q1", "Q2", "Q3", "Q4", "Jan", "Jun", "Dec"], 2)
    data = {"Category": selected}
    for vn in var_names:
        data[vn] = [round(random.uniform(*val_range), 1) for _ in selected]
    df = pd.DataFrame(data)
    return df, f"{title}: {var_names[0]} vs {var_names[1]}", "grouped_bar"


def render_kaggle_chart(df, title, path, chart_type, style_cfg):
    """Render a kaggle-like chart with configurable style."""
    with plt.style.context(style_cfg["style"]):
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
                df.iloc[:, 1], labels=df.iloc[:, 0], autopct='%1.1f%%',
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

        elif chart_type == "grouped_bar":
            cats = df.iloc[:, 0]
            x = np.arange(len(cats))
            n_vars = len(df.columns) - 1
            width = 0.8 / n_vars
            for i, col in enumerate(df.columns[1:]):
                bars = ax.bar(x + i * width, df[col], width, label=col)
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}",
                            ha="center", va="bottom", fontsize=style_cfg["fontsize"]-3)
            ax.set_xticks(x + width * n_vars / 2)
            ax.set_xticklabels(cats.astype(str), rotation=30, fontsize=style_cfg["fontsize"]-2)
            ax.legend(fontsize=style_cfg["fontsize"]-2)

        ax.set_title(title[:60], fontsize=style_cfg["fontsize"]+1)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)
    return True


def collect_kaggle_like(limit=3000):
    """Generate diverse Kaggle-like tabular charts."""
    out_dir = OUTPUT_BASE / "kaggle_like"
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    done = set()
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["slug"])
                except:
                    pass

    collected = []
    errors = 0
    out_f = open(manifest_path, "a")
    random.seed(42)
    np.random.seed(42)

    generators = [
        ("cat", _gen_kaggle_categorical, 0.4),
        ("ts", _gen_kaggle_timeseries, 0.3),
        ("mv", _gen_kaggle_multivar, 0.3),
    ]

    for i in tqdm(range(limit), desc="Kaggle-like", unit="chart"):
        slug = f"kaggle_{i:05d}"
        if slug in done:
            continue
        try:
            # Weighted random generator selection
            gen_type = random.choices(
                [g[0] for g in generators],
                weights=[g[2] for g in generators]
            )[0]
            gen_fn = next(g[1] for g in generators if g[0] == gen_type)
            df, title, chart_type = gen_fn(i)

            if len(df) < 2:
                errors += 1
                continue

            csv_path = out_dir / "tables" / f"{slug}.csv"
            df.to_csv(csv_path, index=False)

            style_cfg = _rand_style()
            img_path = out_dir / "png" / f"{slug}.png"
            ok = render_kaggle_chart(df, title, img_path, chart_type, style_cfg)
            if not ok:
                errors += 1
                continue

            meta = {
                "slug": slug,
                "title": title,
                "source": "kaggle_like",
                "chart_type": chart_type,
                "gen_type": gen_type,
                "style": style_cfg["style"],
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
    print(f"  Kaggle-like: {len(collected)} new + {len(done)} existing, errors={errors}")
    return collected


# ═══════════════════════════════════════════
# Scientific: Academic-style charts
# ═══════════════════════════════════════════

SCIENTIFIC_DOMAINS = [
    "Physics", "Chemistry", "Biology", "Medicine", "Environmental Science",
    "Materials Science", "Astronomy", "Geology", "Neuroscience", "Ecology",
]

SCIENTIFIC_PATTERNS = [
    "exponential_growth", "exponential_decay", "sigmoid", "gaussian",
    "linear_regression", "polynomial", "oscillation", "power_law",
    "log_scale", "step_function",
]


def _gen_scientific_data(idx):
    """Generate academic-style scientific data."""
    pattern = SCIENTIFIC_PATTERNS[idx % len(SCIENTIFIC_PATTERNS)]
    domain = SCIENTIFIC_DOMAINS[idx % len(SCIENTIFIC_DOMAINS)]
    n = random.randint(8, 25)
    noise_level = random.uniform(0.02, 0.15)

    x = np.linspace(0, 10, n)

    if pattern == "exponential_growth":
        rate = random.uniform(0.2, 0.5)
        y = np.exp(rate * x) * random.uniform(1, 10)
        x_label, y_label = "Time (hours)", "Concentration (μM)"
        title = f"{domain}: Exponential Growth Kinetics"

    elif pattern == "exponential_decay":
        half_life = random.uniform(1, 5)
        y = 100 * np.exp(-0.693 / half_life * x)
        x_label, y_label = "Time (min)", "Activity (%)"
        title = f"{domain}: Decay Profile (t½={half_life:.1f} min)"

    elif pattern == "sigmoid":
        k = random.uniform(0.5, 2)
        x0 = random.uniform(3, 7)
        y = 100 / (1 + np.exp(-k * (x - x0)))
        x_label, y_label = "Dose (mg/kg)", "Response (%)"
        title = f"{domain}: Dose-Response Curve"

    elif pattern == "gaussian":
        mu = random.uniform(3, 7)
        sigma = random.uniform(0.5, 2)
        y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) * random.uniform(50, 200)
        x_label, y_label = "Wavelength (nm)", "Intensity (a.u.)"
        title = f"{domain}: Spectral Peak Analysis"

    elif pattern == "linear_regression":
        slope = random.uniform(0.5, 5)
        intercept = random.uniform(-10, 10)
        y = slope * x + intercept
        x_label, y_label = "Concentration (mM)", "Absorbance"
        title = f"{domain}: Calibration Curve (R²≈{random.uniform(0.92, 0.99):.2f})"

    elif pattern == "polynomial":
        a, b, c = random.uniform(-0.5, 0.5), random.uniform(-2, 2), random.uniform(0, 50)
        y = a * x**2 + b * x + c
        x_label, y_label = "Temperature (°C)", "Viscosity (cP)"
        title = f"{domain}: Temperature-Viscosity Relationship"

    elif pattern == "oscillation":
        freq = random.uniform(0.5, 3)
        amp = random.uniform(10, 100)
        y = amp * np.sin(freq * x) * np.exp(-0.1 * x)
        x_label, y_label = "Time (s)", "Amplitude (mV)"
        title = f"{domain}: Damped Oscillation"

    elif pattern == "power_law":
        exponent = random.uniform(0.3, 2.5)
        y = random.uniform(1, 10) * (x + 0.1) ** exponent
        x_label, y_label = "Size (μm)", "Frequency"
        title = f"{domain}: Power Law Distribution (α={exponent:.1f})"

    elif pattern == "log_scale":
        y = np.log10(x + 1) * random.uniform(10, 100)
        x_label, y_label = "Iterations", "Loss"
        title = f"{domain}: Training Convergence"

    else:  # step_function
        steps = sorted(random.sample(range(1, 10), 3))
        y = np.zeros_like(x)
        for s in steps:
            y += (x >= s) * random.uniform(5, 30)
        x_label, y_label = "Voltage (V)", "Current (mA)"
        title = f"{domain}: Step Response"

    # Add noise
    y_noisy = y + np.random.randn(n) * np.abs(y).mean() * noise_level
    y_noisy = np.round(y_noisy, 3)
    x_rounded = np.round(x, 2)

    df = pd.DataFrame({x_label: x_rounded, y_label: y_noisy})
    return df, title, pattern, x_label, y_label


def render_scientific_chart(df, title, path, pattern, x_label, y_label, style_cfg):
    """Render academic-style scientific chart."""
    with plt.style.context(style_cfg["style"]):
        fig, ax = plt.subplots(figsize=style_cfg["figsize"])
        _apply_style(ax, style_cfg)

        x, y = df.iloc[:, 0], df.iloc[:, 1]
        chart_type = random.choice(["scatter_line", "scatter", "line", "bar_scientific"])

        if chart_type == "scatter_line":
            ax.scatter(x, y, s=30, zorder=5, edgecolors="black", linewidths=0.5)
            ax.plot(x, y, alpha=0.5, linewidth=1)

        elif chart_type == "scatter":
            ax.scatter(x, y, s=40, edgecolors="black", linewidths=0.5)
            # Add error bars
            yerr = np.abs(y) * random.uniform(0.03, 0.1)
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="gray", alpha=0.5, capsize=2)

        elif chart_type == "line":
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.5)
            # Annotate min/max
            imin, imax = y.idxmin(), y.idxmax()
            ax.annotate(f"{y[imin]:.2f}", (x[imin], y[imin]),
                       textcoords="offset points", xytext=(0, -12),
                       ha="center", fontsize=style_cfg["fontsize"]-3)
            ax.annotate(f"{y[imax]:.2f}", (x[imax], y[imax]),
                       textcoords="offset points", xytext=(0, 10),
                       ha="center", fontsize=style_cfg["fontsize"]-3)

        elif chart_type == "bar_scientific":
            bars = ax.bar(range(len(df)), y, width=0.7)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([f"{v:.1f}" for v in x], rotation=45, fontsize=style_cfg["fontsize"]-3)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=style_cfg["fontsize"]-3)

        ax.set_xlabel(x_label, fontsize=style_cfg["fontsize"])
        ax.set_ylabel(y_label, fontsize=style_cfg["fontsize"])
        ax.set_title(title[:70], fontsize=style_cfg["fontsize"]+1)
        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)
    return chart_type


def collect_scientific(limit=3000):
    """Generate academic-style scientific charts."""
    out_dir = OUTPUT_BASE / "scientific"
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    done = set()
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["slug"])
                except:
                    pass

    collected = []
    errors = 0
    out_f = open(manifest_path, "a")
    random.seed(43)
    np.random.seed(43)

    for i in tqdm(range(limit), desc="Scientific", unit="chart"):
        slug = f"sci_{i:05d}"
        if slug in done:
            continue
        try:
            df, title, pattern, x_label, y_label = _gen_scientific_data(i)
            if len(df) < 3:
                errors += 1
                continue

            csv_path = out_dir / "tables" / f"{slug}.csv"
            df.to_csv(csv_path, index=False)

            style_cfg = _rand_style()
            img_path = out_dir / "png" / f"{slug}.png"
            chart_type = render_scientific_chart(df, title, img_path, pattern, x_label, y_label, style_cfg)

            meta = {
                "slug": slug,
                "title": title,
                "source": "scientific",
                "pattern": pattern,
                "chart_type": chart_type,
                "domain": SCIENTIFIC_DOMAINS[i % len(SCIENTIFIC_DOMAINS)],
                "style": style_cfg["style"],
                "csv_path": str(csv_path),
                "image_path": str(img_path),
                "n_rows": len(df),
            }
            out_f.write(json.dumps(meta, default=str) + "\n")
            out_f.flush()
            collected.append(meta)

        except Exception:
            errors += 1

    out_f.close()
    print(f"  Scientific: {len(collected)} new + {len(done)} existing, errors={errors}")
    return collected


# ═══════════════════════════════════════════
# Additional: Complex charts (subplots, dual-axis, text answers)
# ═══════════════════════════════════════════

COMPARISON_TEMPLATES = [
    ("Country Comparison: {metric}", ["USA", "China", "India", "Germany", "Japan", "Brazil", "UK", "France", "Canada", "Australia"]),
    ("City Rankings: {metric}", ["Tokyo", "Delhi", "Shanghai", "São Paulo", "Mexico City", "Cairo", "Mumbai", "Beijing", "Dhaka", "Osaka"]),
    ("Company Performance: {metric}", ["Apple", "Google", "Microsoft", "Amazon", "Meta", "Tesla", "Samsung", "Intel", "IBM", "Oracle"]),
    ("University Rankings: {metric}", ["MIT", "Stanford", "Harvard", "Oxford", "Cambridge", "Caltech", "ETH Zurich", "UCL", "Chicago", "Princeton"]),
]

COMPARISON_METRICS = [
    ("GDP per Capita ($K)", 10, 80),
    ("Population (M)", 5, 1400),
    ("Revenue ($B)", 10, 400),
    ("Research Output", 1000, 50000),
    ("Satisfaction Score", 60, 99),
    ("Growth Rate (%)", -5, 25),
    ("Market Share (%)", 1, 40),
    ("Efficiency Index", 50, 100),
]


def _gen_additional_comparison(idx):
    """Charts designed for text-answer questions (which is highest/lowest, etc.)."""
    template = COMPARISON_TEMPLATES[idx % len(COMPARISON_TEMPLATES)]
    metric = COMPARISON_METRICS[idx % len(COMPARISON_METRICS)]
    title_template, entities = template
    metric_name, lo, hi = metric

    n = random.randint(4, min(8, len(entities)))
    selected = random.sample(entities, n)
    values = sorted([round(random.uniform(lo, hi), 1) for _ in selected], reverse=True)
    # Shuffle to make it non-obvious
    combined = list(zip(selected, values))
    random.shuffle(combined)
    selected, values = zip(*combined)

    df = pd.DataFrame({"Name": list(selected), metric_name: list(values)})
    title = title_template.format(metric=metric_name.split("(")[0].strip())
    return df, title


def _gen_additional_dual_axis(idx):
    """Dual-axis chart with two different scales."""
    n = random.randint(6, 12)
    years = list(range(2013, 2013 + n))
    metric1_name = random.choice(["Revenue ($M)", "Users (M)", "Production (tons)", "Spending ($B)"])
    metric2_name = random.choice(["Growth (%)", "Efficiency (%)", "Margin (%)", "Satisfaction (%)"])
    v1 = np.cumsum(np.random.uniform(5, 30, n)).round(1)
    v2 = np.random.uniform(40, 95, n).round(1)
    df = pd.DataFrame({"Year": years, metric1_name: v1, metric2_name: v2})
    title = f"{metric1_name.split('(')[0].strip()} and {metric2_name.split('(')[0].strip()} Over Time"
    return df, title, metric1_name, metric2_name


def _gen_additional_stacked(idx):
    """Stacked bar chart."""
    categories = random.sample(["Q1", "Q2", "Q3", "Q4", "Jan", "Feb", "Mar", "Apr", "May", "Jun"], random.randint(4, 6))
    segments = random.sample(["Product A", "Product B", "Product C", "Product D", "Service", "Other"], random.randint(2, 4))
    data = {"Period": categories}
    for seg in segments:
        data[seg] = [round(random.uniform(10, 100), 1) for _ in categories]
    df = pd.DataFrame(data)
    title = f"Revenue Breakdown by Segment ({random.randint(2022, 2025)})"
    return df, title, segments


def render_additional_chart(df, title, path, chart_subtype, style_cfg, **kwargs):
    """Render complex additional charts."""
    with plt.style.context(style_cfg["style"]):
        if chart_subtype == "comparison":
            fig, ax = plt.subplots(figsize=style_cfg["figsize"])
            _apply_style(ax, style_cfg)
            colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
            bars = ax.barh(df.iloc[:, 0].astype(str), df.iloc[:, 1], color=colors)
            for bar in bars:
                w = bar.get_width()
                ax.text(w, bar.get_y() + bar.get_height()/2, f" {w:.1f}",
                        ha="left", va="center", fontsize=style_cfg["fontsize"]-2)
            ax.set_xlabel(df.columns[1], fontsize=style_cfg["fontsize"])
            ax.set_title(title[:60], fontsize=style_cfg["fontsize"]+1)

        elif chart_subtype == "dual_axis":
            fig, ax1 = plt.subplots(figsize=style_cfg["figsize"])
            _apply_style(ax1, style_cfg)
            m1, m2 = kwargs["metric1"], kwargs["metric2"]
            color1, color2 = "#1f77b4", "#ff7f0e"
            ax1.bar(df.iloc[:, 0].astype(str), df[m1], color=color1, alpha=0.7, label=m1)
            ax1.set_ylabel(m1, color=color1, fontsize=style_cfg["fontsize"])
            ax1.tick_params(axis="y", labelcolor=color1)
            ax2 = ax1.twinx()
            ax2.plot(df.iloc[:, 0].astype(str), df[m2], color=color2, marker="o", linewidth=2, label=m2)
            ax2.set_ylabel(m2, color=color2, fontsize=style_cfg["fontsize"])
            ax2.tick_params(axis="y", labelcolor=color2)
            ax1.set_title(title[:60], fontsize=style_cfg["fontsize"]+1)
            plt.xticks(rotation=45, fontsize=style_cfg["fontsize"]-2)
            ax = ax1

        elif chart_subtype == "stacked":
            fig, ax = plt.subplots(figsize=style_cfg["figsize"])
            _apply_style(ax, style_cfg)
            segments = kwargs["segments"]
            x = np.arange(len(df))
            bottom = np.zeros(len(df))
            for seg in segments:
                bars = ax.bar(x, df[seg], bottom=bottom, label=seg)
                bottom += df[seg].values
            ax.set_xticks(x)
            ax.set_xticklabels(df.iloc[:, 0].astype(str), rotation=30, fontsize=style_cfg["fontsize"]-2)
            ax.legend(fontsize=style_cfg["fontsize"]-2)
            ax.set_title(title[:60], fontsize=style_cfg["fontsize"]+1)

        fig.tight_layout()
        fig.savefig(path, dpi=style_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)
    return True


def collect_additional(limit=2000):
    """Generate complex structured charts (comparison, dual-axis, stacked)."""
    out_dir = OUTPUT_BASE / "additional"
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    done = set()
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["slug"])
                except:
                    pass

    collected = []
    errors = 0
    out_f = open(manifest_path, "a")
    random.seed(44)
    np.random.seed(44)

    for i in tqdm(range(limit), desc="Additional", unit="chart"):
        slug = f"add_{i:05d}"
        if slug in done:
            continue
        try:
            style_cfg = _rand_style()
            chart_subtype = random.choices(
                ["comparison", "dual_axis", "stacked"],
                weights=[0.5, 0.25, 0.25]
            )[0]

            render_kwargs = {}
            if chart_subtype == "comparison":
                df, title = _gen_additional_comparison(i)
            elif chart_subtype == "dual_axis":
                df, title, m1, m2 = _gen_additional_dual_axis(i)
                render_kwargs = {"metric1": m1, "metric2": m2}
            else:  # stacked
                df, title, segments = _gen_additional_stacked(i)
                render_kwargs = {"segments": segments}

            if len(df) < 2:
                errors += 1
                continue

            csv_path = out_dir / "tables" / f"{slug}.csv"
            df.to_csv(csv_path, index=False)

            img_path = out_dir / "png" / f"{slug}.png"
            ok = render_additional_chart(df, title, img_path, chart_subtype, style_cfg, **render_kwargs)
            if not ok:
                errors += 1
                continue

            meta = {
                "slug": slug,
                "title": title,
                "source": "additional",
                "chart_subtype": chart_subtype,
                "style": style_cfg["style"],
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
    print(f"  Additional: {len(collected)} new + {len(done)} existing, errors={errors}")
    return collected


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Chart generation v3: new sources for v8")
    parser.add_argument("--source", choices=["kaggle_like", "scientific", "additional", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Override per-source limit")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    total = []

    if args.source in ("kaggle_like", "all"):
        total.extend(collect_kaggle_like(args.limit or 3000))

    if args.source in ("scientific", "all"):
        total.extend(collect_scientific(args.limit or 3000))

    if args.source in ("additional", "all"):
        total.extend(collect_additional(args.limit or 2000))

    print(f"\nTOTAL v3: {len(total)} charts generated")


if __name__ == "__main__":
    main()
