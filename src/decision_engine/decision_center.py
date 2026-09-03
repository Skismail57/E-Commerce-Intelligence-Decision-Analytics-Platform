from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class DecisionEngine:
    """Automated business intelligence + recommendation center.

    Aggregates outputs from:
        - Anomaly alerts
        - Churn predictions
        - Inventory health
        - Product matrix
        - Marketing campaign ROAS
    and produces:
        1. Executive alerts (Critical / High / Medium / Low priority)
        2. Ranked, specific, actionable recommendations
        3. Estimated impact in ₹ revenue / profit and customer count
        4. Decision center summary CSV (consumable by Power BI / Streamlit)
    """

    PRIORITY_WEIGHTS = {"CRITICAL": 100, "HIGH": 50, "MEDIUM": 20, "LOW": 5, "INFO": 1}

    def __init__(self, data_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else settings.STAGING_DATA_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROCESSED_DATA_DIR
        settings.ensure_dirs()

    def _read_csv(self, name: str, required: bool = False) -> Optional[pd.DataFrame]:
        path = self.processed_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            for dc in [c for c in df.columns if "date" in c.lower()]:
                try:
                    df[dc] = pd.to_datetime(df[dc])
                except Exception:
                    pass
            return df
        if required:
            logger.warning(f"Missing required processed artifact: {name}")
        return None

    def _generate_executive_alerts(self, anomaly_alerts: Optional[pd.DataFrame],
                                    churn_preds: Optional[pd.DataFrame],
                                    inventory_health: Optional[pd.DataFrame],
                                    product_matrix: Optional[pd.DataFrame],
                                    campaign_perf: Optional[pd.DataFrame]) -> List[Dict]:
        alerts: List[Dict] = []

        if anomaly_alerts is not None and len(anomaly_alerts) > 0:
            sev_mask = anomaly_alerts["severity"].isin(["critical", "high"]) if "severity" in anomaly_alerts.columns else pd.Series(False)
            critical_anoms = anomaly_alerts[sev_mask] if len(anomaly_alerts) else anomaly_alerts.iloc[0:0]
            for _, r in critical_anoms.head(5).iterrows():
                kpi = str(r.get("kpi", "KPI"))
                chg = float(r.get("change_pct", 0.0))
                sev = str(r.get("severity", "MEDIUM")).upper()
                title = (
                    f"🚨 ANOMALY: {kpi} {('▲' if chg > 0 else '▼')} {abs(chg):.1f}% on {r.get('date','')}"
                )
                alerts.append({
                    "alert_id": f"BIZ-A-{len(alerts)+1:04d}",
                    "category": "ANOMALY",
                    "priority": sev if sev in self.PRIORITY_WEIGHTS else "HIGH",
                    "priority_score": self.PRIORITY_WEIGHTS.get(sev, 20),
                    "title": title,
                    "detail": f"Possible causes: {r.get('possible_causes','')}. Value: ₹{r.get('value',0):,.0f}, Baseline: ₹{r.get('baseline_28d',0):,.0f}",
                    "impact_revenue_inr": round(abs(float(r.get("value", 0.0)) - float(r.get("baseline_28d", 0.0))), 0),
                    "impact_customers": 0,
                })

        if churn_preds is not None and len(churn_preds) > 0:
            high_risk = churn_preds[churn_preds["risk_tier"] == "High"] if "risk_tier" in churn_preds.columns else None
            if high_risk is not None and len(high_risk) > 0:
                clv_col = [c for c in ["clv_inr", "clv", "clv_value", "lifetime_value"] if c in high_risk.columns]
                avg_clv = float(high_risk[clv_col[0]].median()) if clv_col else 15000.0
                spend_col = [c for c in ["total_spend", "spend_inr"] if c in high_risk.columns]
                if spend_col:
                    avg_clv = float(high_risk[spend_col[0]].median()) if high_risk[spend_col[0]].notna().any() else avg_clv
                count = len(high_risk)
                hv_count = 0
                if "clv_tier" in high_risk.columns:
                    hv_count = int(high_risk["clv_tier"].isin(["High", "Medium-High"]).sum())
                alerts.append({
                    "alert_id": f"BIZ-A-{len(alerts)+1:04d}",
                    "category": "CUSTOMER",
                    "priority": "HIGH" if count > 500 else "MEDIUM",
                    "priority_score": self.PRIORITY_WEIGHTS["HIGH"] if count > 500 else self.PRIORITY_WEIGHTS["MEDIUM"],
                    "title": f"🎯 CUSTOMER RETENTION: {count:,} high-value customers showing HIGH churn risk",
                    "detail": (
                        f"{count:,} customers churn probability ≥ 50%. "
                        f"Top-risk segments: {hv_count:,} high/medium-high CLV tier customers. "
                        "Recommended retention campaign via email/SMS + personalized coupons."
                    ),
                    "impact_revenue_inr": round(count * avg_clv * 0.4, 0),
                    "impact_customers": count,
                })

        if inventory_health is not None and len(inventory_health) > 0:
            stock_out_mask = inventory_health["stock_status"].isin(["Out of Stock", "Critical", "Reorder"]) if "stock_status" in inventory_health.columns else pd.Series(False)
            at_risk = inventory_health[stock_out_mask]
            units_demand_col = [c for c in ["demand_90d", "avg_daily_demand"] if c in at_risk.columns]
            price_col = [c for c in ["selling_price", "unit_price"] if c in at_risk.columns]
            if len(at_risk) > 0:
                rev_exposure = 0.0
                if units_demand_col and price_col:
                    rev_exposure = float((at_risk[units_demand_col[0]].fillna(0) * at_risk[price_col[0]].fillna(0)).sum()) / 90 * 14
                alerts.append({
                    "alert_id": f"BIZ-A-{len(alerts)+1:04d}",
                    "category": "INVENTORY",
                    "priority": "CRITICAL" if len(at_risk) > 200 else "HIGH",
                    "priority_score": self.PRIORITY_WEIGHTS["CRITICAL"] if len(at_risk) > 200 else self.PRIORITY_WEIGHTS["HIGH"],
                    "title": f"⚠️ INVENTORY: {len(at_risk):,} SKUs at risk of stock-out within 14 days",
                    "detail": (
                        f"Reorder priority triggered for {len(at_risk):,} products. "
                        "Expected 14-day revenue at risk; automated PO generation recommended for Critical products."
                    ),
                    "impact_revenue_inr": round(rev_exposure, 0),
                    "impact_customers": 0,
                })

        if product_matrix is not None and len(product_matrix) > 0:
            remove_mask = product_matrix["quadrant"].str.lower() == "remove" if "quadrant" in product_matrix.columns else None
            if remove_mask is not None and remove_mask.any():
                remove_count = int(remove_mask.sum())
                rev_col = [c for c in ["total_revenue_inr", "revenue_inr"] if c in product_matrix.columns]
                profit_col = [c for c in ["gross_profit_inr", "profit"] if c in product_matrix.columns]
                remove_rev = float(product_matrix.loc[remove_mask, rev_col[0]].sum()) if rev_col else 0
                remove_profit = float(product_matrix.loc[remove_mask, profit_col[0]].sum()) if profit_col else 0
                alerts.append({
                    "alert_id": f"BIZ-A-{len(alerts)+1:04d}",
                    "category": "PRODUCT",
                    "priority": "MEDIUM",
                    "priority_score": self.PRIORITY_WEIGHTS["MEDIUM"],
                    "title": f"📦 PRODUCT RATIONALIZATION: {remove_count:,} SKUs classified REMOVE (low revenue + low margin)",
                    "detail": (
                        f"These {remove_count:,} products generated ₹{remove_rev/1e5:,.1f}L revenue but only "
                        f"₹{remove_profit/1e5:,.1f}L gross profit. Consider SKU consolidation, clearance sale, or discontinued."
                    ),
                    "impact_revenue_inr": round(remove_rev * 0.10, 0),
                    "impact_customers": 0,
                })

        if campaign_perf is not None and len(campaign_perf) > 0:
            roas_col = [c for c in ["roas", "roas_ratio"] if c in campaign_perf.columns]
            if roas_col:
                below_1 = campaign_perf[campaign_perf[roas_col[0]] < 1.0]
                if len(below_1) > 0:
                    spend_col = [c for c in ["spend_inr", "total_spend", "budget_inr"] if c in campaign_perf.columns]
                    wasted = float(below_1[spend_col[0]].sum()) if spend_col else 0.0
                    alerts.append({
                        "alert_id": f"BIZ-A-{len(alerts)+1:04d}",
                        "category": "MARKETING",
                        "priority": "HIGH" if wasted > 5e5 else "MEDIUM",
                        "priority_score": self.PRIORITY_WEIGHTS["HIGH"] if wasted > 5e5 else self.PRIORITY_WEIGHTS["MEDIUM"],
                        "title": f"📢 MARKETING: {len(below_1)} campaigns with ROAS < 1x (negative return)",
                        "detail": (
                            f"Combined under-performing spend ₹{wasted/1e5:,.1f}L. "
                            "Recommended: Pause below-1 ROAS channels, reallocate budget to top-5 performing campaigns, revise audience targeting."
                        ),
                        "impact_revenue_inr": round(wasted * 0.5, 0),
                        "impact_customers": 0,
                    })
        return alerts

    def _generate_recommendations(self, alerts: List[Dict]) -> pd.DataFrame:
        recs: List[Dict] = []
        ranked_alerts = sorted(alerts, key=lambda a: (-a["priority_score"], -a["impact_revenue_inr"]))
        action_templates = {
            "ANOMALY": [
                ("Launch forensic review", 0.4, "Category/channel root-cause drill-down", "24h"),
                ("Deploy ops team", 0.25, "Monitor next 3 days, reforecast if needed", "48h"),
                ("Notify senior leadership", 0.15, "Slack/email briefing packet", "1h"),
            ],
            "CUSTOMER": [
                ("Launch retention campaign", 0.35, "Personalized 15% off for High-risk customers", "3 days"),
                ("1:1 outreach for High CLV", 0.30, "Account manager outreach for top 100 CLV High-risk customers", "1 week"),
                ("Loyalty program refresh", 0.15, "Surprise points or tier bump for Medium-risk", "2 weeks"),
            ],
            "INVENTORY": [
                ("Raise emergency PO", 0.50, "Auto-generate PO for 2× safety stock on Critical", "48h"),
                ("Supplier escalation", 0.25, "Priority call to at-risk suppliers, expedite delivery", "24h"),
                ("Shift marketing spend", 0.10, "Reduce promo spend on out-of-stock products", "1 week"),
            ],
            "PRODUCT": [
                ("Clearance sale", 0.40, "End-of-season 25% off + bundling for REMOVE quadrant", "2 weeks"),
                ("SKU consolidation", 0.25, "Merge low-performing SKUs into hero variants", "1 month"),
                ("Supplier renegotiation", 0.15, "Target 7% cost reduction on Volume/Premium", "1 month"),
            ],
            "MARKETING": [
                ("Pause low-ROAS channels", 0.45, "Reallocate 60% of budget → top 3 performing campaigns", "Immediate"),
                ("A/B test creatives", 0.25, "New hero creative test for CTR < 2%", "2 weeks"),
                ("Audience retargeting", 0.20, "Launch cart-abandon retargeting for drop-off > 60%", "1 week"),
            ],
        }
        for alert in ranked_alerts:
            cat = alert["category"]
            templates = action_templates.get(cat, [("General review", 0.10, "Manual investigation", "1 week")])
            for i, (action, impact_factor, detail, timeline) in enumerate(templates, 1):
                recs.append({
                    "rec_id": f"REC-{alert['alert_id']}-{i}",
                    "alert_id": alert["alert_id"],
                    "category": cat,
                    "priority": alert["priority"],
                    "priority_score": alert["priority_score"],
                    "recommendation": f"{alert['title'][:80]} → {action}",
                    "action_details": detail,
                    "timeline": timeline,
                    "estimated_impact_pct": round(impact_factor * 100, 0),
                    "estimated_revenue_impact_inr": round(alert["impact_revenue_inr"] * impact_factor, 0),
                    "estimated_customers_impact": round(alert["impact_customers"] * impact_factor),
                    "owner_mapping": {
                        "ANOMALY": "Head of Operations + Data Analyst",
                        "CUSTOMER": "Customer Success + Marketing",
                        "INVENTORY": "Operations + Procurement",
                        "PRODUCT": "Merchandising + Category Manager",
                        "MARKETING": "Digital Marketing Lead",
                    }.get(cat, "TBD"),
                    "status": "OPEN",
                })
        return pd.DataFrame(recs)

    def run_all(self, staging_dir: Optional[Path] = None, processed_dir: Optional[Path] = None,
                save: bool = True) -> Dict:
        self.data_dir = Path(staging_dir) if staging_dir else self.data_dir
        self.processed_dir = Path(processed_dir) if processed_dir else self.processed_dir
        settings.ensure_dirs()

        anomaly_alerts = self._read_csv("anomaly_alerts", required=False)
        churn_preds = self._read_csv("churn_predictions", required=False)
        inventory_health = self._read_csv("inventory_health_aggregated", required=False)
        if inventory_health is None:
            inventory_health = self._read_csv("inventory_health", required=False)
        product_matrix = self._read_csv("product_matrix", required=False)
        campaign_perf = self._read_csv("campaign_performance", required=False)

        executive_alerts = self._generate_executive_alerts(
            anomaly_alerts, churn_preds, inventory_health, product_matrix, campaign_perf,
        )
        executive_alerts_df = pd.DataFrame(executive_alerts).sort_values(
            ["priority_score", "impact_revenue_inr"], ascending=[False, False]
        ).reset_index(drop=True) if executive_alerts else pd.DataFrame()

        recommendations_df = self._generate_recommendations(executive_alerts)

        summary: Dict = {
            "alert_count_by_category": (executive_alerts_df["category"].value_counts().to_dict()
                                        if "category" in executive_alerts_df.columns else {}),
            "alert_count_by_priority": (executive_alerts_df["priority"].value_counts().to_dict()
                                        if "priority" in executive_alerts_df.columns else {}),
            "top_3_alerts": executive_alerts_df.head(3).to_dict(orient="records") if len(executive_alerts_df) else [],
            "total_open_recommendations": len(recommendations_df),
            "recommendations_by_category": (recommendations_df["category"].value_counts().to_dict()
                                           if "category" in recommendations_df.columns else {}),
        }
        if "estimated_revenue_impact_inr" in recommendations_df.columns:
            summary["total_potential_revenue_impact_inr_cr"] = round(
                float(recommendations_df["estimated_revenue_impact_inr"].sum()) / 1e7, 2
            )
        if "estimated_customers_impact" in recommendations_df.columns:
            summary["total_customers_retained_estimate"] = int(recommendations_df["estimated_customers_impact"].sum())
        summary_df = pd.DataFrame([summary])

        outputs: Dict[str, pd.DataFrame] = {
            "decision_center_executive_alerts": executive_alerts_df,
            "decision_center_recommendations": recommendations_df,
            "decision_center_summary": summary_df,
        }
        if save:
            for name, df in outputs.items():
                if df is not None and len(df) > 0:
                    out = self.processed_dir / f"{name}.csv"
                    df.to_csv(out, index=False)
                    logger.info(f"Saved {name} -> {out}")

        summary["outputs"] = {k: v.shape for k, v in outputs.items() if v is not None}
        return summary
