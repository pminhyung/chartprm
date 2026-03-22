#!/usr/bin/env python
"""
Chart generation v2: CSV preprocessing + value labels + quality filters.

Track A: OWID/WorldBank with prepare_csv_for_charting
Track B: Synthetic additional generation

Usage:
  python scripts/gen_charts_v2.py --source owid --limit 3000
  python scripts/gen_charts_v2.py --source worldbank --limit 1000
  python scripts/gen_charts_v2.py --source synthetic --limit 3000
  python scripts/gen_charts_v2.py --source all
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

BASE = Path("/ex_disk2/mhpark/poc/chartvr")
OUTPUT_BASE = BASE / "data" / "charts_v2"


# ═══════════════════════════════════════════
# CSV Preprocessing
# ═══════════════════════════════════════════

def prepare_csv_for_charting(df: pd.DataFrame, slug: str = "") -> tuple:
    """차트로 렌더링 가능한 크기와 스케일로 CSV 정리.
    Returns (df, scale_info) where scale_info has unit suffixes."""
    scale_info = {}

    # Detect entity/year columns
    entity_col = None
    year_col = None
    for col in df.columns:
        cl = col.lower()
        if cl in ("entity", "country", "economy", "region", "name"):
            entity_col = col
        if cl in ("year", "date", "time"):
            year_col = col

    # 1. Entity 수 제한: 상위 5-8개만
    if entity_col and df[entity_col].nunique() > 8:
        n = random.choice([5, 6, 7, 8])
        # Pick entities with most data points
        top = df[entity_col].value_counts().head(n).index.tolist()
        df = df[df[entity_col].isin(top)]

    # 2. 연도 범위 제한: 최근 10-15년
    if year_col:
        try:
            years = pd.to_numeric(df[year_col], errors="coerce").dropna()
            if len(years) > 0:
                max_year = int(years.max())
                span = int(years.max() - years.min())
                if span > 15:
                    keep_years = range(max_year - random.randint(10, 15), max_year + 1)
                    df = df[pd.to_numeric(df[year_col], errors="coerce").isin(keep_years)]
        except Exception:
            pass

    # 3. Row limit
    if len(df) > 20:
        df = df.tail(20)

    # 4. 값 스케일 정규화
    for col in df.select_dtypes(include=[np.number]).columns:
        max_val = df[col].abs().max()
        if pd.isna(max_val):
            continue
        if max_val > 1_000_000:
            df[col] = df[col] / 1_000_000
            scale_info[col] = "(millions)"
        elif max_val > 10_000:
            df[col] = df[col] / 1_000
            scale_info[col] = "(thousands)"

    # 5. Round values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].round(2)

    # Drop rows with all NaN numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df = df.dropna(subset=numeric_cols, how="all")

    return df, scale_info


# ═══════════════════════════════════════════
# Chart Rendering (with value labels)
# ═══════════════════════════════════════════

def render_chart_with_labels(df: pd.DataFrame, title: str, path: Path, scale_info: dict = None):
    """Render chart with value labels on all data points."""
    fig, ax = plt.subplots(figsize=(10, 6))
    scale_info = scale_info or {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        plt.close(fig)
        return False

    label_col = df.columns[0]
    has_year = any(c.lower() in ("year", "date") for c in df.columns)

    if has_year:
        year_col = [c for c in df.columns if c.lower() in ("year", "date")][0]
        val_cols = [c for c in numeric_cols if c != year_col]

        if not val_cols:
            plt.close(fig)
            return False

        # Check if there are multiple entities (grouped by entity column)
        entity_col = None
        for c in df.columns:
            if c.lower() in ("entity", "country", "economy", "region", "name"):
                entity_col = c
                break

        if entity_col and df[entity_col].nunique() > 1:
            # Multi-entity line chart
            for entity in df[entity_col].unique():
                sub = df[df[entity_col] == entity].sort_values(year_col)
                for vc in val_cols[:1]:  # Use first value column
                    line = ax.plot(sub[year_col].astype(str), sub[vc], marker="o", markersize=4, label=str(entity))
                    for x, y in zip(sub[year_col].astype(str), sub[vc]):
                        if pd.notna(y):
                            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                                       xytext=(0, 8), ha="center", fontsize=6)
        else:
            # Single entity line/bar
            for vc in val_cols[:3]:
                ax.plot(df[year_col].astype(str), df[vc], marker="o", markersize=4, label=vc)
                for x, y in zip(df[year_col].astype(str), df[vc]):
                    if pd.notna(y):
                        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                                   xytext=(0, 8), ha="center", fontsize=7)

        ax.legend(fontsize=7, loc="best")
        plt.xticks(rotation=45, fontsize=8)
        suffix = scale_info.get(val_cols[0], "") if val_cols else ""
        if suffix:
            ax.set_ylabel(f"{val_cols[0]} {suffix}", fontsize=9)

    elif len(numeric_cols) == 1 and len(df) <= 15:
        # Bar chart
        bars = ax.bar(df[label_col].astype(str), df[numeric_cols[0]])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)
        suffix = scale_info.get(numeric_cols[0], "")
        ax.set_ylabel(f"{numeric_cols[0]} {suffix}" if suffix else numeric_cols[0], fontsize=9)
        plt.xticks(rotation=30, fontsize=8)

    elif len(numeric_cols) >= 2:
        # Grouped bar
        x = np.arange(len(df))
        n_cols = min(len(numeric_cols), 4)
        width = 0.8 / n_cols
        for i, col in enumerate(numeric_cols[:n_cols]):
            bars = ax.bar(x + i * width, df[col], width, label=col)
            for bar in bars:
                h = bar.get_height()
                if pd.notna(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h,
                            f"{h:.1f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x + width * n_cols / 2)
        ax.set_xticklabels(df[label_col].astype(str), rotation=30, fontsize=8)
        ax.legend(fontsize=7)

    else:
        # Horizontal bar
        bars = ax.barh(df[label_col].astype(str), df[numeric_cols[0]])
        for bar in bars:
            w = bar.get_width()
            ax.text(w, bar.get_y() + bar.get_height() / 2,
                    f" {w:.1f}", ha="left", va="center", fontsize=8)

    ax.set_title(title[:60], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


# ═══════════════════════════════════════════
# OWID Collection v2
# ═══════════════════════════════════════════

def collect_owid_v2(limit: int = 3000):
    """OWID with CSV preprocessing + value labels."""
    out_dir = OUTPUT_BASE / "owid"
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    # Get slugs
    slugs = []
    try:
        resp = requests.get("https://ourworldindata.org/sitemap.xml", timeout=30)
        if resp.ok:
            slugs = list(set(re.findall(r"ourworldindata\.org/grapher/([a-z0-9-]+)", resp.text)))
    except Exception:
        pass

    if not slugs:
        print("Failed to get OWID slugs")
        return []

    random.seed(42)
    random.shuffle(slugs)
    selected = slugs[:limit]
    print(f"OWID: processing {len(selected)} slugs", flush=True)

    # Resume
    done = set()
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["slug"])
                except:
                    pass
    pending = [s for s in selected if s not in done]
    print(f"  Done: {len(done)}, Pending: {len(pending)}", flush=True)

    collected = []
    errors = 0
    out_f = open(manifest_path, "a")

    for slug in tqdm(pending, desc="OWID v2", unit="chart"):
        try:
            csv_url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
            csv_resp = requests.get(csv_url, timeout=30)
            if csv_resp.status_code != 200:
                errors += 1
                continue

            df_raw = pd.read_csv(pd.io.common.StringIO(csv_resp.text))
            if len(df_raw) < 3 or len(df_raw.columns) < 2:
                errors += 1
                continue

            # Preprocess
            df, scale_info = prepare_csv_for_charting(df_raw, slug)
            if len(df) < 3:
                errors += 1
                continue

            # Check complexity after preprocessing
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                errors += 1
                continue
            max_val = df[numeric_cols].abs().max().max()
            if pd.isna(max_val) or max_val > 10000:
                errors += 1
                continue

            # Save CSV
            csv_path = out_dir / "tables" / f"{slug}.csv"
            df.to_csv(csv_path, index=False)

            # Get title
            try:
                config_resp = requests.get(f"https://ourworldindata.org/grapher/{slug}.config.json", timeout=10)
                title = config_resp.json().get("title", slug.replace("-", " ").title()) if config_resp.ok else slug.replace("-", " ").title()
            except:
                title = slug.replace("-", " ").title()

            # Render with labels
            img_path = out_dir / "png" / f"{slug}.png"
            ok = render_chart_with_labels(df, title, img_path, scale_info)
            if not ok:
                errors += 1
                continue

            meta = {
                "slug": slug,
                "title": title,
                "source": "owid",
                "csv_path": str(csv_path),
                "image_path": str(img_path),
                "n_rows": len(df),
                "n_numeric_cols": len(numeric_cols),
                "max_value": float(max_val),
            }
            out_f.write(json.dumps(meta, default=str) + "\n")
            out_f.flush()
            collected.append(meta)

        except Exception:
            errors += 1

    out_f.close()
    print(f"  OWID v2: {len(collected)} new + {len(done)} existing, errors={errors}", flush=True)
    return collected


# ═══════════════════════════════════════════
# WorldBank Collection v2
# ═══════════════════════════════════════════

def collect_worldbank_v2(limit: int = 1000):
    """WorldBank with CSV preprocessing + value labels."""
    import wbgapi as wb

    out_dir = OUTPUT_BASE / "worldbank"
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    popular = [
        "SP.POP.TOTL", "NY.GDP.MKTP.CD", "NY.GDP.PCAP.CD", "SP.DYN.LE00.IN",
        "SE.ADT.LITR.ZS", "SH.DYN.MORT", "EN.ATM.CO2E.PC", "IT.NET.USER.ZS",
        "SL.UEM.TOTL.ZS", "FP.CPI.TOTL.ZG", "NE.EXP.GNFS.ZS", "SP.URB.TOTL.IN.ZS",
        "EG.USE.ELEC.KH.PC", "AG.LND.FRST.ZS", "SH.XPD.CHEX.GD.ZS",
    ]
    try:
        all_ind = [ind["id"] for ind in wb.series.list() if "id" in ind]
        random.seed(42)
        random.shuffle(all_ind)
        indicators = popular + all_ind[:limit]
        indicators = list(dict.fromkeys(indicators))[:limit]
    except:
        indicators = popular

    countries = ["USA", "CHN", "IND", "GBR", "DEU", "JPN", "BRA"]  # 7 countries

    # Resume
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

    for ind_id in tqdm(indicators[:limit], desc="WB v2", unit="ind"):
        slug = ind_id.replace(".", "_").lower()
        if slug in done:
            continue
        try:
            df_raw = wb.data.DataFrame(ind_id, economy=countries, time=range(2010, 2024))
            if df_raw.empty:
                errors += 1
                continue

            df_raw = df_raw.reset_index()
            df, scale_info = prepare_csv_for_charting(df_raw, slug)
            if len(df) < 3:
                errors += 1
                continue

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                errors += 1
                continue
            max_val = df[numeric_cols].abs().max().max()
            if pd.isna(max_val) or max_val > 10000:
                errors += 1
                continue

            csv_path = out_dir / "tables" / f"{slug}.csv"
            df.to_csv(csv_path, index=False)

            try:
                info = wb.series.get(ind_id)
                title = info.get("value", ind_id)[:60]
            except:
                title = ind_id

            img_path = out_dir / "png" / f"{slug}.png"
            ok = render_chart_with_labels(df, title, img_path, scale_info)
            if not ok:
                errors += 1
                continue

            meta = {
                "slug": slug,
                "title": title,
                "source": "worldbank",
                "csv_path": str(csv_path),
                "image_path": str(img_path),
                "n_rows": len(df),
                "max_value": float(max_val),
            }
            out_f.write(json.dumps(meta, default=str) + "\n")
            out_f.flush()
            collected.append(meta)

        except Exception:
            errors += 1

    out_f.close()
    print(f"  WB v2: {len(collected)} new + {len(done)} existing, errors={errors}", flush=True)
    return collected


# ═══════════════════════════════════════════
# Synthetic v2 (additional)
# ═══════════════════════════════════════════

def collect_synthetic_v2(limit: int = 3000):
    """Additional synthetic charts with value labels."""
    sys.path.insert(0, str(BASE / "scripts"))
    from chart_qa_data_pipeline import CHART_TYPES, _generate_synthetic_data

    out_dir = OUTPUT_BASE / "synthetic"
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    per_type = max(limit // len(CHART_TYPES), 1)

    # Resume
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
    start_idx = 100  # Offset from v1 (which used 0-32)

    for chart_type in tqdm(CHART_TYPES, desc="Synthetic v2", unit="type"):
        for i in range(start_idx, start_idx + per_type):
            slug = f"{chart_type}_{i:03d}"
            if slug in done:
                continue
            try:
                df, title = _generate_synthetic_data(chart_type, i)
                # Apply preprocessing
                df, scale_info = prepare_csv_for_charting(df, slug)
                if len(df) < 3:
                    errors += 1
                    continue

                csv_path = out_dir / "tables" / f"{slug}.csv"
                df.to_csv(csv_path, index=False)

                img_path = out_dir / "png" / f"{slug}.png"
                ok = render_chart_with_labels(df, title, img_path, scale_info)
                if not ok:
                    errors += 1
                    continue

                meta = {
                    "slug": slug,
                    "chart_type": chart_type,
                    "title": title,
                    "source": "synthetic",
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
    print(f"  Synthetic v2: {len(collected)} new + {len(done)} existing, errors={errors}", flush=True)
    return collected


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["owid", "worldbank", "synthetic", "all"], default="all")
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    total = []

    if args.source in ("owid", "all"):
        total.extend(collect_owid_v2(args.limit if args.source == "owid" else 3000))

    if args.source in ("worldbank", "all"):
        total.extend(collect_worldbank_v2(args.limit if args.source == "worldbank" else 1000))

    if args.source in ("synthetic", "all"):
        total.extend(collect_synthetic_v2(args.limit if args.source == "synthetic" else 3000))

    print(f"\nTOTAL v2: {len(total)} charts", flush=True)


if __name__ == "__main__":
    main()
