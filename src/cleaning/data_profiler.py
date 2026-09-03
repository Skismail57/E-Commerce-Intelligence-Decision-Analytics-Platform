from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import json
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger(__name__)


class DataProfiler:
    def __init__(self):
        self.reports: Dict[str, Dict] = {}
        self.kpis: Optional[Dict] = None

    def profile_df(self, df: pd.DataFrame, name: str = "DataFrame") -> Dict:
        logger.info(f"Profiling {name}: {len(df):,} rows x {len(df.columns)} cols")

        total_cells = len(df) * len(df.columns)
        total_nulls = int(df.isnull().sum().sum())
        dups = int(df.duplicated().sum())

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        numeric_summary = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            numeric_summary[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "zeros": int((s == 0).sum()),
                "negatives": int((s < 0).sum()),
                "nulls": int(df[col].isnull().sum()),
            }

        categorical_summary = {}
        for col in categorical_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            val_counts = s.value_counts()
            categorical_summary[col] = {
                "unique": int(s.nunique()),
                "nulls": int(df[col].isnull().sum()),
                "top_values": val_counts.head(5).to_dict(),
            }

        report = {
            "name": name,
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "null_count": total_nulls,
            "null_pct": round((total_nulls / total_cells) * 100, 2) if total_cells > 0 else 0.0,
            "duplicates": dups,
            "duplicate_pct": round((dups / len(df)) * 100, 2) if len(df) > 0 else 0.0,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "date_cols": date_cols,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }

        self.reports[name] = report
        return report

    def profile_all(
        self, data_dir: Path, tables: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        data_dir = Path(data_dir)
        all_csvs = sorted(data_dir.glob("*.csv"))
        results = {}
        for csv_file in all_csvs:
            table_name = csv_file.stem
            if tables and table_name not in tables:
                continue
            df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
            for dc in [c for c in df.columns if "date" in c.lower()]:
                try:
                    df[dc] = pd.to_datetime(df[dc])
                except Exception:
                    pass
            results[table_name] = self.profile_df(df, name=table_name)
        logger.info(f"Profiled {len(results)} tables from {data_dir}")
        return results

    def get_summary(self) -> pd.DataFrame:
        rows = []
        for name, r in self.reports.items():
            rows.append({
                "table": name,
                "rows": r["rows"],
                "cols": r["cols"],
                "null_pct": r["null_pct"],
                "duplicate_pct": r["duplicate_pct"],
                "memory_mb": r["memory_mb"],
                "numeric_cols": len(r["numeric_cols"]),
                "cat_cols": len(r["categorical_cols"]),
                "date_cols": len(r["date_cols"]),
            })
        return pd.DataFrame(rows)

    def profile_with_kpis(self, data_dir: Path) -> Dict:
        data_dir = Path(data_dir)
        self.profile_all(data_dir)

        summary_df = self.get_summary()
        total_rows = int(summary_df["rows"].sum()) if len(summary_df) > 0 else 0
        total_cols = int(summary_df["cols"].sum()) if len(summary_df) > 0 else 0
        total_tables = len(self.reports)
        total_cells = total_rows * total_cols
        total_nulls = sum(r["null_count"] for r in self.reports.values())
        overall_null_pct = round((total_nulls / total_cells) * 100, 2) if total_cells > 0 else 0.0
        total_memory_mb = round(summary_df["memory_mb"].sum(), 2) if len(summary_df) > 0 else 0.0

        summary = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_tables": total_tables,
            "overall_null_pct": overall_null_pct,
            "memory_mb": total_memory_mb,
        }

        kpis = {
            "estimated_revenue": "N/A",
            "total_orders": "N/A",
            "total_customers": "N/A",
            "total_products": "N/A",
            "avg_order_value": "N/A",
            "gross_margin_pct": "N/A",
        }

        dfs = {}
        for table_name in ["orders", "order_items", "customers", "products"]:
            csv_path = data_dir / f"{table_name}.csv"
            if csv_path.exists():
                try:
                    dfs[table_name] = pd.read_csv(csv_path, low_memory=False)
                except Exception:
                    dfs[table_name] = None

        if "orders" in dfs and dfs["orders"] is not None:
            orders_df = dfs["orders"]
            kpis["total_orders"] = int(len(orders_df))
            valid_statuses = {"Delivered", "Returned", "Shipped"}
            if "order_status" in orders_df.columns and "order_total" in orders_df.columns:
                valid_orders = orders_df[orders_df["order_status"].isin(valid_statuses)]
                revenue = float(valid_orders["order_total"].sum()) if len(valid_orders) > 0 else 0.0
                kpis["estimated_revenue"] = round(revenue, 2)
                if len(valid_orders) > 0:
                    kpis["avg_order_value"] = round(revenue / len(valid_orders), 2)

        if "customers" in dfs and dfs["customers"] is not None:
            kpis["total_customers"] = int(len(dfs["customers"]))

        if "products" in dfs and dfs["products"] is not None:
            kpis["total_products"] = int(len(dfs["products"]))

        if ("order_items" in dfs and dfs["order_items"] is not None and
            "products" in dfs and dfs["products"] is not None):
            order_items_df = dfs["order_items"]
            products_df = dfs["products"]
            try:
                if ("quantity" in order_items_df.columns and
                    "product_id" in order_items_df.columns and
                    "unit_price" in order_items_df.columns and
                    "cost_price" in products_df.columns and
                    "product_id" in products_df.columns):
                    merged = order_items_df.merge(
                        products_df[["product_id", "cost_price"]],
                        on="product_id",
                        how="left",
                    )
                    revenue_total = float((merged["quantity"] * merged["unit_price"]).sum())
                    cost_total = float((merged["quantity"] * merged["cost_price"].fillna(0)).sum())
                    if revenue_total > 0:
                        margin = ((revenue_total - cost_total) / revenue_total) * 100
                        kpis["gross_margin_pct"] = round(margin, 2)
            except Exception:
                pass

        self.kpis = kpis

        return {
            "summary": summary,
            "kpis": kpis,
            "tables": self.reports,
        }


def _derive_kpis_from_profiler(profiler: DataProfiler) -> Dict:
    kpis = {
        "estimated_revenue": "N/A",
        "total_orders": "N/A",
        "total_customers": "N/A",
        "total_products": "N/A",
        "avg_order_value": "N/A",
        "gross_margin_pct": "N/A",
    }

    if hasattr(profiler, "kpis") and profiler.kpis is not None:
        return profiler.kpis

    reports = profiler.reports
    if "orders" in reports:
        kpis["total_orders"] = reports["orders"]["rows"]
    if "customers" in reports:
        kpis["total_customers"] = reports["customers"]["rows"]
    if "products" in reports:
        kpis["total_products"] = reports["products"]["rows"]

    return kpis


def _format_value(val) -> str:
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, float):
        if abs(val) >= 1_000_000:
            return f"${val / 1_000_000:,.2f}M"
        elif abs(val) >= 1_000:
            return f"${val:,.2f}"
        else:
            return f"${val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _build_html(profiler: DataProfiler) -> str:
    reports = profiler.reports
    summary_df = profiler.get_summary()

    total_rows = int(summary_df["rows"].sum()) if len(summary_df) > 0 else 0
    total_cols = int(summary_df["cols"].sum()) if len(summary_df) > 0 else 0
    total_tables = len(reports)
    total_cells = total_rows * total_cols
    total_nulls = sum(r["null_count"] for r in reports.values())
    overall_null_pct = round((total_nulls / total_cells) * 100, 2) if total_cells > 0 else 0.0
    total_memory_mb = round(summary_df["memory_mb"].sum(), 2) if len(summary_df) > 0 else 0.0

    kpis = _derive_kpis_from_profiler(profiler)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    table_cards_html = ""
    for table_name, report in reports.items():
        num_summary = report["numeric_summary"]
        cat_summary = report["categorical_summary"]

        num_rows_html = ""
        if num_summary:
            for col, stats in num_summary.items():
                num_rows_html += f"""
                <tr>
                    <td class="td-left">{col}</td>
                    <td>{stats['min']:,.4f}</td>
                    <td>{stats['max']:,.4f}</td>
                    <td>{stats['mean']:,.4f}</td>
                    <td>{stats['median']:,.4f}</td>
                    <td>{stats['std']:,.4f}</td>
                    <td>{stats['p25']:,.4f}</td>
                    <td>{stats['p75']:,.4f}</td>
                    <td>{stats['zeros']:,}</td>
                    <td>{stats['negatives']:,}</td>
                    <td>{stats['nulls']:,}</td>
                </tr>"""
        else:
            num_rows_html = '<tr><td colspan="11" class="text-center">No numeric columns</td></tr>'

        cat_rows_html = ""
        if cat_summary:
            for col, stats in cat_summary.items():
                top_vals = ", ".join(
                    [f"{k} ({v:,})" for k, v in stats["top_values"].items()]
                )
                cat_rows_html += f"""
                <tr>
                    <td class="td-left">{col}</td>
                    <td>{stats['unique']:,}</td>
                    <td>{stats['nulls']:,}</td>
                    <td class="td-left">{top_vals}</td>
                </tr>"""
        else:
            cat_rows_html = '<tr><td colspan="4" class="text-center">No categorical columns</td></tr>'

        table_cards_html += f"""
        <div class="table-card">
            <div class="table-header">
                <h3 class="table-name">{table_name}</h3>
                <div class="table-meta">
                    <span class="meta-chip">{report['rows']:,} rows</span>
                    <span class="meta-chip">{report['cols']:,} cols</span>
                    <span class="meta-chip null">Null: {report['null_pct']}%</span>
                    <span class="meta-chip dup">Dup: {report['duplicate_pct']}%</span>
                    <span class="meta-chip mem">{report['memory_mb']} MB</span>
                </div>
            </div>

            <div class="section-subtitle">Numeric Summary</div>
            <div class="table-wrap">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>Std</th>
                            <th>P25</th>
                            <th>P75</th>
                            <th>Zeros</th>
                            <th>Neg</th>
                            <th>Nulls</th>
                        </tr>
                    </thead>
                    <tbody>
                        {num_rows_html}
                    </tbody>
                </table>
            </div>

            <div class="section-subtitle">Categorical Top Values</div>
            <div class="table-wrap">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Unique</th>
                            <th>Nulls</th>
                            <th>Top 5 Values</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cat_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E-Commerce Intelligence Platform — Data Profiling Report</title>
<style>
:root {{
    --bg-primary: #0d0d1a;
    --bg-secondary: #16162a;
    --bg-card: #1e1e38;
    --bg-card-hover: #252547;
    --border-subtle: #2d2d55;
    --text-primary: #f0eefc;
    --text-secondary: #b5b3d4;
    --text-muted: #7a77a0;
    --accent-purple: #a855f7;
    --accent-pink: #ec4899;
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --gradient-main: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    --gradient-alt: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
    --gradient-header: linear-gradient(135deg, #4f46e5 0%, #9333ea 40%, #db2777 100%);
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --info: #3b82f6;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}}
.container {{ max-width: 1400px; margin: 0 auto; padding: 0 24px 40px; }}
.report-header {{
    background: var(--gradient-header);
    padding: 56px 24px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid var(--border-subtle);
}}
.report-header::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 140%;
    height: 200%;
    background: radial-gradient(circle at 20% 50%, rgba(168,85,247,0.25) 0%, transparent 50%),
                radial-gradient(circle at 80% 50%, rgba(236,72,153,0.2) 0%, transparent 50%);
    pointer-events: none;
}}
.report-header h1 {{
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    position: relative;
    background: linear-gradient(135deg, #ffffff 0%, #e9d5ff 50%, #fbcfe8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}}
.report-header p {{
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.95rem;
    position: relative;
}}
.section {{
    margin-top: 36px;
}}
.section-title {{
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.section-title::before {{
    content: '';
    width: 4px;
    height: 24px;
    background: var(--gradient-main);
    border-radius: 2px;
}}
.section-subtitle {{
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 20px 0 12px;
}}
.exec-summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
}}
.summary-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}}
.summary-card:hover {{
    transform: translateY(-2px);
    border-color: var(--accent-purple);
}}
.summary-card::after {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-main);
    opacity: 0;
    transition: opacity 0.2s;
}}
.summary-card:hover::after {{ opacity: 1; }}
.summary-label {{
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 8px;
    font-weight: 600;
}}
.summary-value {{
    font-size: 1.8rem;
    font-weight: 700;
    background: var(--gradient-alt);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
}}
.kpi-card {{
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 22px;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    border-color: var(--accent-violet);
    box-shadow: 0 12px 40px rgba(139, 92, 246, 0.15);
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 120px;
    height: 120px;
    background: radial-gradient(circle, rgba(168,85,247,0.12) 0%, transparent 70%);
    border-radius: 0 0 0 100%;
}}
.kpi-icon {{
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: var(--gradient-main);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
    font-size: 1.1rem;
}}
.kpi-label {{
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
    font-weight: 600;
}}
.kpi-value {{
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text-primary);
    position: relative;
}}
.kpi-value.na {{
    color: var(--text-muted);
    font-size: 1.3rem;
    font-weight: 500;
}}
.table-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    transition: border-color 0.2s;
}}
.table-card:hover {{
    border-color: var(--accent-purple);
}}
.table-header {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-subtle);
}}
.table-name {{
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
}}
.table-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}}
.meta-chip {{
    font-size: 0.75rem;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 20px;
    background: rgba(99, 102, 241, 0.12);
    color: #c4b5fd;
    border: 1px solid rgba(99, 102, 241, 0.25);
}}
.meta-chip.null {{
    background: rgba(245, 158, 11, 0.1);
    color: #fcd34d;
    border-color: rgba(245, 158, 11, 0.2);
}}
.meta-chip.dup {{
    background: rgba(239, 68, 68, 0.1);
    color: #fca5a5;
    border-color: rgba(239, 68, 68, 0.2);
}}
.meta-chip.mem {{
    background: rgba(16, 185, 129, 0.1);
    color: #6ee7b7;
    border-color: rgba(16, 185, 129, 0.2);
}}
.table-wrap {{
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
    margin-bottom: 8px;
}}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}}
.data-table thead {{
    background: rgba(139, 92, 246, 0.12);
}}
.data-table th {{
    padding: 12px 14px;
    text-align: right;
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border-subtle);
    white-space: nowrap;
}}
.data-table th:first-child,
.data-table td.td-left,
.data-table th.td-left {{
    text-align: left;
}}
.data-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid rgba(45, 45, 85, 0.5);
    text-align: right;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}}
.data-table td.td-left {{
    color: var(--accent-purple);
    font-weight: 500;
    white-space: normal;
    max-width: 300px;
}}
.data-table tbody tr:hover {{
    background: rgba(139, 92, 246, 0.05);
}}
.data-table tbody tr:last-child td {{
    border-bottom: none;
}}
.text-center {{
    text-align: center !important;
    color: var(--text-muted) !important;
    padding: 20px !important;
}}
.report-footer {{
    text-align: center;
    margin-top: 48px;
    padding: 28px 24px;
    border-top: 1px solid var(--border-subtle);
    color: var(--text-muted);
    font-size: 0.85rem;
}}
.report-footer span {{
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
}}
@media (max-width: 768px) {{
    .report-header h1 {{ font-size: 1.5rem; }}
    .summary-value {{ font-size: 1.4rem; }}
    .kpi-value {{ font-size: 1.3rem; }}
    .container {{ padding: 0 16px 32px; }}
    .data-table {{ font-size: 0.75rem; }}
    .data-table th, .data-table td {{ padding: 8px 10px; }}
}}
</style>
</head>
<body>
<div class="report-header">
    <h1>E-Commerce Intelligence Platform — Data Profiling Report</h1>
    <p>Comprehensive data quality assessment &amp; KPI preview</p>
</div>

<div class="container">
    <div class="section">
        <h2 class="section-title">Executive Summary</h2>
        <div class="exec-summary-grid">
            <div class="summary-card">
                <div class="summary-label">Total Rows</div>
                <div class="summary-value">{total_rows:,}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Total Columns</div>
                <div class="summary-value">{total_cols:,}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Total Tables</div>
                <div class="summary-value">{total_tables:,}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Overall Null %</div>
                <div class="summary-value">{overall_null_pct}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Memory Usage</div>
                <div class="summary-value">{total_memory_mb} MB</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">KPI Preview</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">&#128176;</div>
                <div class="kpi-label">Estimated Revenue</div>
                <div class="kpi-value {"na" if kpis["estimated_revenue"] == "N/A" else ""}">{_format_value(kpis["estimated_revenue"])}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">&#128203;</div>
                <div class="kpi-label">Total Orders</div>
                <div class="kpi-value {"na" if kpis["total_orders"] == "N/A" else ""}">{_format_value(kpis["total_orders"])}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">&#128101;</div>
                <div class="kpi-label">Total Customers</div>
                <div class="kpi-value {"na" if kpis["total_customers"] == "N/A" else ""}">{_format_value(kpis["total_customers"])}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">&#128230;</div>
                <div class="kpi-label">Total Products</div>
                <div class="kpi-value {"na" if kpis["total_products"] == "N/A" else ""}">{_format_value(kpis["total_products"])}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">&#128200;</div>
                <div class="kpi-label">Avg Order Value</div>
                <div class="kpi-value {"na" if kpis["avg_order_value"] == "N/A" else ""}">{_format_value(kpis["avg_order_value"])}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">&#128202;</div>
                <div class="kpi-label">Gross Margin</div>
                <div class="kpi-value {"na" if kpis["gross_margin_pct"] == "N/A" else ""}">{kpis["gross_margin_pct"] if kpis["gross_margin_pct"] == "N/A" else f"{kpis['gross_margin_pct']}%"}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">Table-by-Table Detail</h2>
        {table_cards_html}
    </div>
</div>

<div class="report-footer">
    Generated on <span>{generated_at}</span> &mdash; E-Commerce Intelligence &amp; Decision Analytics Platform
</div>
</body>
</html>"""
    return html


def export_profile_report(profiler: DataProfiler, output_format: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    format_upper = output_format.lower()
    if format_upper not in {"html", "json", "both"}:
        raise ValueError(f"output_format must be 'html', 'json', or 'both', got '{output_format}'")

    json_path = output_dir / "data_profile_report.json"
    html_path = output_dir / "data_profile_report.html"
    main_path: Optional[Path] = None

    if format_upper in {"json", "both"}:
        summary_df = profiler.get_summary()
        json_payload = {
            "generated_at": datetime.now().isoformat(),
            "reports": profiler.reports,
            "summary": json.loads(summary_df.to_json(orient="records")),
        }
        if hasattr(profiler, "kpis") and profiler.kpis is not None:
            json_payload["kpis"] = profiler.kpis
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, default=str)
        logger.info(f"JSON report saved to {json_path}")
        main_path = json_path

    if format_upper in {"html", "both"}:
        html_content = _build_html(profiler)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML report saved to {html_path}")
        main_path = html_path

    return main_path
