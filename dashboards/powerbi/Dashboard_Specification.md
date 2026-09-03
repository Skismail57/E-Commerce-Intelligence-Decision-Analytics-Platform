# E-Commerce Intelligence & Decision Analytics Platform
## Power BI Dashboard Specification

---

## Overview

This Power BI dashboard provides a comprehensive 360° view of e-commerce performance across 7 specialized pages, designed for executives, operations teams, marketing, and business analysts.

**Data Source:** PostgreSQL Data Warehouse via DirectQuery
**Refresh Schedule:** Daily automatic refresh + on-demand manual refresh
**Security:** Row-level security by region/department

---

## Page 1: Executive Overview

**Target Audience:** C-Suite, Business Leaders
**Purpose:** High-level business health snapshot with key trends

### KPI Cards (Top Row)
- **Total Revenue** (₹ Cr) - with YoY growth %
- **Net Revenue** (₹ Cr) - after returns/cancellations
- **Gross Profit** (₹ Cr) - actual profitability
- **Gross Margin %** - profitability indicator
- **Total Orders** - order volume
- **Unique Customers** - customer base size
- **Average Order Value** (₹) - AOV metric
- **Return Rate %** - operational efficiency

### Visualizations

**Revenue Trend (Line Chart)**
- X-axis: Month (last 12 months)
- Y-axis: Revenue (₹ Cr)
- Lines: Gross Revenue, Net Revenue
- Tooltip: Month, Revenue values, Growth %

**Profit Trend (Area Chart)**
- X-axis: Month
- Y-axis: Gross Profit (₹ Cr)
- Color gradient by margin performance

**Regional Performance (Map/Choropleth)**
- Indian states colored by revenue
- Size bubble by order volume
- Tooltip: State, Revenue, Orders, AOV

**Category Performance (Treemap)**
- Size: Revenue contribution
- Color: Gross Margin %
- Hierarchy: Category → Subcategory

**Key Metrics Table**
- Top 5 performing states by revenue
- Top 5 performing categories
- Bottom 3 categories (action items)

### Data Sources
- `kpi_executive_snapshot` - Executive KPIs with ANALYSIS_AS_OF_DATE (2024-12-31)
- `kpi_monthly_trend` - Monthly trend data from vw_monthly_sales
- `kpi_region_performance` - Regional breakdown from vw_customer_360
- `kpi_category_performance` - Category performance from vw_product_360

---

## Page 2: Sales Intelligence

**Target Audience:** Sales Managers, Business Analysts
**Purpose:** Deep dive into sales performance and trends

### Visualizations

**Daily Revenue Trend (Line Chart)**
- X-axis: Date (last 90 days)
- Y-axis: Daily Revenue (₹ L)
- 7-day moving average overlay
- Annotations for promotions/events

**Order Volume Analysis (Column Chart)**
- X-axis: Day of week
- Y-axis: Average orders
- Color by order status (Delivered, Cancelled, Returned)

**AOV Trend (Combo Chart)**
- X-axis: Month
- Y-axis (left): AOV (₹)
- Y-axis (right): Order count
- Correlation analysis

**Product Performance (Scatter Plot)**
- X-axis: Units Sold
- Y-axis: Revenue (₹)
- Size: Margin %
- Color: Category
- Quadrant lines for Stars/Volume/Remove/Premium

**Regional Heatmap (Matrix)**
- Rows: States
- Columns: Months
- Values: Revenue (color intensity)
- Drill-down to city level

**YoY/MoM Growth (Waterfall Chart)**
- Breakdown of revenue growth/decline
- By category contribution
- Positive/negative color coding

### Data Sources
- `vw_daily_sales`
- `vw_order_line_fact`
- `vw_product_360`
- `kpi_monthly_trend`

---

## Page 3: Customer Intelligence

**Target Audience:** Marketing, Customer Success
**Purpose:** Customer segmentation, retention, and lifetime value analysis

### Visualizations

**RFM Segment Distribution (Donut Chart)**
- Segments: Champions, Loyal Customers, Potential Loyalists, New Customers, At Risk, Can't Lose Them, Lost Customers
- Percentage and count per segment
- Click to filter other visuals

**Customer Cohort Retention (Matrix Heatmap)**
- Rows: Signup month cohorts
- Columns: Months since signup (0-12)
- Values: Retention % (color intensity)
- Pattern analysis for retention trends

**CLV Distribution (Histogram)**
- X-axis: Customer Lifetime Value (₹)
- Y-axis: Customer count
- Pareto line (80/20 rule)
- Top 10% contribution highlight

**New vs Returning Customers (Stacked Area Chart)**
- X-axis: Month
- Y-axis: Customer count
- Layers: New, Returning
- Growth rate annotation

**Customer Tenure Analysis (Box Plot)**
- X-axis: RFM Segment
- Y-axis: Customer tenure (days)
- Distribution by segment
- Outlier identification

**Churn Risk Pipeline (Funnel Chart)**
- Stages: Total → Active → At Risk → Churned
- 30-day, 60-day, 90-day churn rates
- Intervention points highlighted

### Data Sources
- `vw_rfm_segments` - RFM analysis from feature_engineering.py
- `vw_analytics_cohort_retention` - Cohort retention matrix
- `vw_customer_360` - Customer master with fixed aggregation (no fan-out)
- `kpi_customer_lifetime` - CLV predictions from BG/NBD + Gamma-Gamma model
- `kpi_top_10_customers` - High-value customer list

---

## Page 4: Product Intelligence

**Target Audience:** Category Managers, Merchandising
**Purpose:** Product performance, profitability, and lifecycle management

### Visualizations

**Product Matrix (Scatter Plot)**
- X-axis: Revenue (₹)
- Y-axis: Margin %
- Quadrant background: Stars/Volume/Remove/Premium
- Product bubbles sized by units sold
- Click for product details

**Top/Bottom Products (Table)**
- Columns: Product Name, Category, Revenue, Units, Margin %, Return Rate, Rating
- Tabs: Top 20 by Revenue, Bottom 20 by Margin, Highest Returns
- Conditional formatting

**Product Lifecycle (Gauge Chart)**
- Overall portfolio lifecycle distribution
- Stages: Introduction, Growth, Maturity, Decline
- Count and revenue per stage

**Category Performance (Ribbon Chart)**
- X-axis: Categories
- Y-axis: Revenue
- Ribbon thickness: Margin contribution
- Rank changes over time

**Return Analysis (Combo Chart)**
- X-axis: Categories
- Y-axis (left): Return Rate %
- Y-axis (right): Total Returns (count)
- Size bubble: Revenue impact

**Price Elasticity (Scatter Plot)**
- X-axis: Average Price
- Y-axis: Units Sold
- Trend line with elasticity coefficient
- Color by elasticity type (Elastic/Inelastic)

### Data Sources
- `vw_product_360`
- `vw_analytics_product_matrix`
- `kpi_category_performance`

---

## Page 5: Marketing Intelligence

**Target Audience:** Marketing Team, Performance Analysts
**Purpose:** Campaign performance, channel optimization, ROI analysis

### Visualizations

**Marketing Funnel (Funnel Chart)**
- Stages: Impressions → Clicks → Sessions → Product Views → Cart → Checkout → Purchase
- Conversion rates between stages
- Drop-off analysis

**Campaign Performance (Table)**
- Columns: Campaign Name, Channel, Spend, Impressions, Clicks, CTR, Orders, Revenue, CAC, ROAS
- Sorting by ROAS/Revenue
- Status indicators (Active/Completed)

**Channel Comparison (Radar Chart)**
- Axes: CTR, CPC, Conversion Rate, CAC, ROAS, Revenue
- Lines per channel (Google, Meta, Email, SMS, Organic)
- Performance benchmarking

**CAC vs CLV (Scatter Plot)**
- X-axis: CAC (₹)
- Y-axis: CLV (₹)
- Bubbles: Campaigns
- Color: Profitable (CLV > 3×CAC) vs Unprofitable
- Diagonal line for breakeven

**Spend vs Revenue (Combo Chart)**
- X-axis: Date
- Y-axis (left): Daily Spend (₹)
- Y-axis (right): Attributed Revenue (₹)
- Correlation analysis
- Campaign period highlights

**Channel Attribution (Stacked Bar Chart)**
- X-axis: Month
- Y-axis: Revenue
- Stack: Marketing channels
- Organic vs Paid split

### Data Sources
- `vw_marketing_channel_funnel`
- `kpi_payment_methods`
- Marketing campaign tables

---

## Page 6: Inventory & Operations

**Target Audience:** Operations, Supply Chain, Warehouse Managers
**Purpose:** Inventory optimization, stock-out prevention, supplier performance

### Visualizations

**Inventory Health Status (Card Grid)**
- Cards: Total SKUs, Out of Stock, Critical, Reorder, Healthy, Overstock
- Color coding: Red (critical), Yellow (warning), Green (healthy)
- Click to filter product list

**Stock-Out Risk (Table)**
- Columns: Product, Category, Current Stock, Daily Demand, Days of Stock, Reorder Qty, Urgency
- Filter: Days of Stock < 7
- Sort by urgency
- Action buttons for reorder

**Supplier Performance (Ribbon Chart)**
- X-axis: Suppliers
- Y-axis: Revenue contribution
- Ribbon: Reliability score
- Color: Return rate

**Inventory Turnover (Line Chart)**
- X-axis: Month
- Y-axis: Turnover ratio
- Lines by category
- Industry benchmark line

**Store Performance (Map)**
- Store locations with performance bubbles
- Size: Revenue
- Color: Inventory efficiency
- Tooltip: Store metrics

**Demand Forecast vs Actual (Line Chart)**
- X-axis: Date (next 30 days)
- Y-axis: Units
- Lines: Forecast, Actual (historical), Confidence interval
- Forecast accuracy metrics

### Data Sources
- `vw_inventory_analytics`
- `vw_inventory_health`
- `vw_supplier_performance`
- `vw_store_performance`
- `vw_demand_daily`

---

## Page 7: Decision Center ⭐

**Target Audience:** All Stakeholders
**Purpose:** Automated business recommendations and alert management

### Alert Sections (Collapsible)

**🚨 Business Alerts**
- Revenue drop > 10% (week-over-week)
- Category performance anomalies
- Order volume spikes/drops
- Payment failure rate spikes
- Visual: Alert timeline with severity levels

**⚠️ Inventory Alerts**
- High-demand products at stock-out risk (< 7 days)
- Overstocked items (> 90 days inventory)
- Supplier delivery delays
- Visual: Risk matrix (Impact × Urgency)

**🎯 Customer Alerts**
- High-value customers showing churn signals
- RFM segment migration (Champions → At Risk)
- Negative review spikes
- Visual: At-risk customer list with intervention suggestions

**💡 Smart Recommendations**
- **Inventory**: "Replenish Electronics category - 12 SKUs at risk"
- **Pricing**: "Consider price increase for Premium quadrant products"
- **Marketing**: "Launch retention campaign for 4,821 at-risk customers"
- **Operations**: "Review supplier X - delivery delays increasing"
- Visual: Prioritized recommendation cards with impact scores

### Decision Support Tools

**What-If Analyzer**
- Slider: Discount % impact on revenue
- Slider: Marketing spend impact on acquisition
- Real-time projection updates

**Scenario Planning**
- Pre-built scenarios: Holiday season, Supply chain disruption, New product launch
- Custom scenario builder

### Data Sources
- `vw_anomaly_prep` - Anomaly detection results from anomaly_detector.py
- `vw_ml_churn_features` - Churn features with temporal split (observation: Jan-Sep 2024, prediction: Oct-Dec 2024)
- Decision engine recommendations from recommendation system
- All KPI views for context with ANALYSIS_AS_OF_DATE (2024-12-31)

---

## Data Model

### Core Tables (DirectQuery)
- `vw_customer_360` - Customer master with metrics
- `vw_product_360` - Product master with metrics  
- `vw_order_line_fact` - Denormalized transaction fact
- `vw_daily_sales` - Daily aggregated sales
- `vw_inventory_health` - Inventory status
- `vw_rfm_segments` - RFM analysis
- `vw_marketing_channel_funnel` - Marketing metrics

### Relationships
```
customers (1) ----< (N) orders
orders (1) ----< (N) order_items
products (1) ----< (N) order_items
categories (1) ----< (N) products
suppliers (1) ----< (N) products
stores (1) ----< (N) orders
marketing_campaigns (1) ----< (N) orders
```

---

## Performance Optimization

### DirectQuery Best Practices
- Use aggregate tables for high-level KPIs
- Limit visual complexity on Executive Overview
- Implement incremental refresh for historical data
- Use query reduction parameters for date filters

### Measures (DAX)
Key calculated measures:
```dax
// Revenue Growth
Revenue Growth % = 
DIVIDE(
    [Net Revenue] - CALCULATE([Net Revenue], SAMEPERIODLASTYEAR(Date[Date])),
    CALCULATE([Net Revenue], SAMEPERIODLASTYEAR(Date[Date]))
)

// Gross Margin
Gross Margin % = 
DIVIDE([Gross Profit], [Total Revenue])

// CLV
Customer Lifetime Value = 
[Average Order Value] * [Purchase Frequency] * 36 * [Gross Margin %]

// Churn Rate
Churn Rate % = 
DIVIDE([Churned Customers], [Total Customers])
```

---

## Security & Governance

### Row-Level Security (RLS)
- **Regional Managers**: Filter by assigned states
- **Category Managers**: Filter by product categories
- **Executives**: Full access
- **External Partners**: Restricted access (aggregated data only)

### Data Refresh
- **Automatic**: Daily 6:00 AM UTC
- **Manual**: On-demand button
- **Failure Alerts**: Email to data team
- **Refresh History**: Log in Power BI service

---

## User Adoption

### Training Materials
- Executive summary (2-page brief)
- Page-specific user guides
- Video tutorials (5 min per page)
- FAQ document

### Feedback Loop
- In-app feedback button
- Monthly usage analytics review
- Quarterly enhancement planning
- Stakeholder interviews

---

## Future Enhancements

### Phase 2 Features
- Real-time data streaming (operational dashboard)
- Natural language Q&A integration
- Mobile-optimized layouts
- Advanced AI-driven insights
- Predictive what-if scenarios

### Integration Roadmap
- ERP system integration for real-time inventory
- CRM integration for customer journey mapping
- Social media sentiment analysis
- Competitor price monitoring

---

## Appendix

### Color Palette
- **Primary**: #7b2ff7 (Purple)
- **Success**: #00c853 (Green)
- **Warning**: #ffab00 (Amber)
- **Danger**: #d32f2f (Red)
- **Neutral**: #f5f5f5 (Light Gray)

### Font Hierarchy
- **Headers**: Segoe UI Bold
- **Body**: Segoe UI Regular
- **Numbers**: Segoe UI Semibold
- **Labels**: Segoe UI Light

### Glossary
- **AOV**: Average Order Value
- **CAC**: Customer Acquisition Cost
- **CLV**: Customer Lifetime Value
- **COGS**: Cost of Goods Sold
- **RFM**: Recency, Frequency, Monetary
- **ROAS**: Return on Ad Spend

---

**Document Version:** 1.0  
**Last Updated:** September 2026  
**Owner:** Data Analytics Team  
**Review Cycle:** Quarterly
