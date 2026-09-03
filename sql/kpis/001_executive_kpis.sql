-- ======================================================================
-- EXECUTIVE KPIs  (14 core business metrics)
--
-- Use case:
--   1) Run as-is to get latest KPI snapshot.
--   2) Swap the CTE `scope` to filter by month/quarter/region/category.
-- ======================================================================

SET search_path TO public;

-- Analysis as-of date - should match dataset end date
-- This prevents using CURRENT_DATE which would be incorrect for historical data
DO $$
BEGIN
    PERFORM set_config('app.analysis_as_of_date', '2024-12-31', false);
END $$;

-- ----------------------------------------------------------------------
-- 01) kpi_executive_snapshot
--     A single row answer to "how is the business doing?"
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW kpi_executive_snapshot AS
WITH scope AS (
    -- Default: entire time range. Filter by category/state/customer_segment in subqueries when needed.
    SELECT 1 AS enabled
),
orders_valid AS (
    SELECT o.*
    FROM orders o, scope s
    WHERE o.order_status IN ('Delivered','Shipped','Processing','Returned')
),
orders_net AS (
    SELECT o.*
    FROM orders o, scope s
    WHERE o.order_status IN ('Delivered','Shipped','Processing')
),
returned_value AS (
    SELECT COALESCE(SUM(o.order_total), 0)  AS val,
           COUNT(DISTINCT o.order_id)       AS cnt
    FROM orders o, scope s
    WHERE o.order_status = 'Returned'
),
cancelled_value AS (
    SELECT COALESCE(SUM(o.order_total), 0) AS val,
           COUNT(DISTINCT o.order_id)      AS cnt
    FROM orders o, scope s
    WHERE o.order_status = 'Cancelled'
),
new_vs_returning AS (
    SELECT
        first_order.customer_id,
        CASE WHEN first_order.orders = 1 THEN 'New' ELSE 'Returning' END AS cust_type,
        o.order_total
    FROM (
        SELECT customer_id, COUNT(DISTINCT order_id) AS orders
        FROM orders_valid
        GROUP BY 1
    ) first_order
    JOIN orders_valid o ON o.customer_id = first_order.customer_id
)
SELECT
    -- Revenue
    ROUND(COALESCE(SUM(ov.order_total), 0) / 1e5, 2)                             AS total_revenue_lakhs,
    ROUND(COALESCE(SUM(ov.order_total), 0), 2)                                   AS total_revenue_inr,
    ROUND(
        (COALESCE(SUM(ov.order_total), 0)
            - COALESCE((SELECT val FROM returned_value), 0)
            - COALESCE((SELECT val FROM cancelled_value), 0)
        )
        , 2)                                                                      AS net_revenue_inr,
    -- Margin
    ROUND(
        100.0 * NULLIF(
            COALESCE(SUM(oi.line_total - oi.quantity * p.cost_price), 0), 0)
          / NULLIF(COALESCE(SUM(oi.line_total), 0), 0), 2)                       AS gross_margin_pct,
    ROUND(COALESCE(SUM(oi.line_total - oi.quantity * p.cost_price), 0), 2)        AS gross_profit_inr,
    -- Order volume
    COUNT(DISTINCT ov.order_id)                                                   AS orders,
    COALESCE(SUM(oi.quantity), 0)                                                 AS units_sold,
    -- Customer
    COUNT(DISTINCT ov.customer_id)                                                AS customers,
    COUNT(DISTINCT CASE WHEN nvr.cust_type = 'New' THEN ov.customer_id END)       AS new_customers,
    COUNT(DISTINCT CASE WHEN nvr.cust_type = 'Returning' THEN ov.customer_id END) AS returning_customers,
    -- Unit economics
    ROUND(1.0 * NULLIF(COALESCE(SUM(ov.order_total), 0), 0)
                / NULLIF(COUNT(DISTINCT ov.order_id), 0), 2)                     AS avg_order_value_inr,
    -- Returns
    ROUND(100.0 * NULLIF((SELECT cnt FROM returned_value), 0)
                / NULLIF(COUNT(DISTINCT o_net.order_id)
                      + (SELECT cnt FROM returned_value), 0), 2)                  AS return_rate_pct,
    -- Acquisition
    ROUND(1.0 * NULLIF(ms.total_spend, 0)
                / NULLIF(COUNT(DISTINCT CASE WHEN nvr.cust_type = 'New'
                                          THEN ov.customer_id END), 0), 2)       AS cac_inr,
    ROUND(1.0 * NULLIF(COALESCE(SUM(ov.order_total), 0), 0)
                / NULLIF(ms.total_spend, 0), 2)                                   AS roas,
    -- Conversion
    ROUND(100.0 * NULLIF(ws.total_checkouts, 0)
                / NULLIF(ws.total_sessions, 0), 2)                                 AS conversion_rate_pct,
    -- Churn
    (SELECT COUNT(*) FROM (
        SELECT c.customer_id
        FROM customers c
        LEFT JOIN orders_valid ov2 ON ov2.customer_id = c.customer_id
        GROUP BY c.customer_id
        HAVING MAX(ov2.order_date) IS NULL
            OR current_setting('app.analysis_as_of_date', true)::DATE - MAX(ov2.order_date)::DATE > 90
    ) churned)                                                                    AS churned_customers_90d,
    -- CLV proxy (AOV × freq × lifespan × margin)
    ROUND(
        (1.0 * NULLIF(COALESCE(SUM(ov.order_total), 0), 0)
                    / NULLIF(COUNT(DISTINCT ov.order_id), 0))
      * (1.0 * NULLIF(COUNT(DISTINCT ov.order_id), 0)
                    / NULLIF(COUNT(DISTINCT ov.customer_id), 0))
      * (36.0)    -- lifespan in months approximation
      * (0.01 * NULLIF(
            100.0 * NULLIF(
                COALESCE(SUM(oi.line_total - oi.quantity * p.cost_price), 0), 0)
              / NULLIF(COALESCE(SUM(oi.line_total), 0), 0), 0))
      , 2)                                                                        AS avg_clv_inr
FROM orders_valid ov
LEFT JOIN orders_net o_net      ON o_net.order_id     = ov.order_id
LEFT JOIN order_items oi        ON oi.order_id        = ov.order_id
LEFT JOIN products p            ON p.product_id      = oi.product_id
LEFT JOIN new_vs_returning nvr  ON nvr.customer_id   = ov.customer_id
CROSS JOIN (SELECT COALESCE(SUM(spend_amount), 0) AS total_spend FROM marketing_spend) ms
CROSS JOIN (SELECT COALESCE(SUM(checkout_completed), 0) AS total_checkouts, COUNT(DISTINCT session_id) AS total_sessions FROM website_sessions) ws,
scope s
GROUP BY s.enabled
LIMIT 1;

COMMENT ON VIEW kpi_executive_snapshot IS
'14 KPIs on one row: Revenue, Margin, Orders, AOV, Customers, CAC, ROAS, Conversion, CLV, Churn, Returns.';


-- ----------------------------------------------------------------------
-- 02) kpi_monthly_trend   — Executive KPIs by month (for Power BI trend charts)
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW kpi_monthly_trend AS
SELECT
    DATE_TRUNC('month', o.order_date)::DATE                                    AS month,
    COUNT(DISTINCT o.order_id)                                                 AS orders,
    ROUND(COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                THEN o.order_total ELSE 0 END), 0), 2)                         AS total_revenue_inr,
    ROUND(COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing')
                THEN o.order_total ELSE 0 END), 0), 2)                         AS net_revenue_inr,
    COUNT(DISTINCT o.customer_id)                                              AS unique_customers,
    ROUND(1.0 * NULLIF(
            COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                          THEN o.order_total ELSE 0 END), 0), 0)
          / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)                          AS aov_inr,
    ROUND(100.0 * NULLIF(
            COUNT(DISTINCT CASE WHEN o.order_status = 'Returned' THEN o.order_id END), 0)
          / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)                          AS return_rate_pct,
    COALESCE(SUM(oi.quantity), 0)                                              AS units_sold,
    COUNT(DISTINCT oi.product_id)                                              AS unique_products_sold
FROM orders o
LEFT JOIN order_items oi ON oi.order_id = o.order_id
GROUP BY 1
ORDER BY 1;

COMMENT ON VIEW kpi_monthly_trend IS 'Monthly KPI trend for revenue, orders, AOV, return rate.';


-- ----------------------------------------------------------------------
-- 03) kpi_region_performance  — State-level geography KPI view
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW kpi_region_performance AS
SELECT
    c.state,
    COUNT(DISTINCT o.order_id)                                                 AS orders,
    COUNT(DISTINCT c.customer_id)                                              AS customers,
    ROUND(COALESCE(SUM(o.order_total), 0), 2)                                  AS revenue_inr,
    ROUND(1.0 * NULLIF(COALESCE(SUM(o.order_total), 0), 0)
                / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)                   AS aov_inr,
    COUNT(DISTINCT c.city)                                                     AS cities_reached,
    ROUND(100.0 * NULLIF(
            COUNT(DISTINCT CASE WHEN o.order_status = 'Returned' THEN o.order_id END), 0)
          / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)                          AS return_rate_pct
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
WHERE o.order_status IN ('Delivered','Shipped','Processing','Returned')
GROUP BY 1
ORDER BY revenue_inr DESC;

COMMENT ON VIEW kpi_region_performance IS 'KPI breakdown by state (22 Indian states from generator).';


-- ----------------------------------------------------------------------
-- 04) kpi_category_performance
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW kpi_category_performance AS
WITH category_orders AS (
    SELECT
        p.category_id,
        COUNT(DISTINCT o.order_id) AS orders,
        SUM(oi.line_total) AS revenue_inr,
        SUM(oi.line_total - oi.quantity * p.cost_price) AS gross_profit_inr,
        SUM(oi.quantity) AS units_sold
    FROM products p
    JOIN order_items oi ON oi.product_id = p.product_id
    JOIN orders o ON o.order_id = oi.order_id
    WHERE o.order_status IN ('Delivered','Shipped','Processing','Returned')
    GROUP BY p.category_id
),
category_returns AS (
    SELECT
        p.category_id,
        COUNT(DISTINCT r.return_id) AS return_count
    FROM products p
    LEFT JOIN returns r ON r.product_id = p.product_id
    GROUP BY p.category_id
),
category_reviews AS (
    SELECT
        p.category_id,
        AVG(rv.rating) AS avg_rating
    FROM products p
    LEFT JOIN reviews rv ON rv.product_id = p.product_id
    GROUP BY p.category_id
)
SELECT
    cat.category_name,
    cat.subcategory,
    co.orders,
    ROUND(COALESCE(co.revenue_inr, 0), 2) AS revenue_inr,
    ROUND(100.0 * NULLIF(co.gross_profit_inr, 0) / NULLIF(co.revenue_inr, 0), 2) AS gross_margin_pct,
    ROUND(COALESCE(co.gross_profit_inr, 0), 2) AS gross_profit_inr,
    COALESCE(co.units_sold, 0) AS units_sold,
    COUNT(DISTINCT p.product_id) AS skus,
    ROUND(100.0 * NULLIF(cr.return_count, 0) / NULLIF(co.units_sold, 0), 2) AS return_rate_pct,
    ROUND(COALESCE(crv.avg_rating, 0), 2) AS avg_rating
FROM categories cat
JOIN products p ON p.category_id = cat.category_id
LEFT JOIN category_orders co ON co.category_id = cat.category_id
LEFT JOIN category_returns cr ON cr.category_id = cat.category_id
LEFT JOIN category_reviews crv ON crv.category_id = cat.category_id
GROUP BY 1, 2, co.orders, co.revenue_inr, co.gross_profit_inr, co.units_sold, cr.return_count, crv.avg_rating
ORDER BY COALESCE(co.revenue_inr, 0) DESC;

COMMENT ON VIEW kpi_category_performance IS 'KPI breakdown by category × subcategory: revenue, margin, returns, rating.';


-- ----------------------------------------------------------------------
-- 05) kpi_customer_lifetime  — CLV cohort view
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW kpi_customer_lifetime AS
WITH cohort AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', signup_date)::DATE                                 AS signup_month,
        customer_segment
    FROM customers
),
cust_ltv AS (
    SELECT
        c.customer_id,
        c.signup_month,
        c.customer_segment,
        COUNT(DISTINCT o.order_id)                                              AS total_orders,
        COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                THEN o.order_total ELSE 0 END), 0)                              AS lifetime_spend,
        ROUND(1.0 * NULLIF(
            COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                          THEN o.order_total ELSE 0 END), 0), 0)
          / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)                          AS aov,
        (MAX(o.order_date)::DATE - MIN(o.order_date)::DATE)                     AS lifespan_days,
        COUNT(DISTINCT CASE WHEN o.order_status = 'Returned' THEN o.order_id END) AS returns
    FROM cohort c
    LEFT JOIN orders o ON o.customer_id = c.customer_id
    GROUP BY 1, 2, 3
)
SELECT
    signup_month,
    customer_segment,
    COUNT(customer_id)                                                          AS cohort_size,
    ROUND(AVG(lifetime_spend), 2)                                              AS avg_ltv_inr,
    ROUND(AVG(total_orders), 2)                                                AS avg_purchase_frequency,
    ROUND(AVG(aov), 2)                                                         AS avg_aov_inr,
    ROUND(AVG(lifespan_days), 1)                                               AS avg_lifespan_days,
    ROUND(100.0 * NULLIF(SUM(returns), 0)
                / NULLIF(SUM(total_orders), 0), 2)                             AS cohort_return_rate_pct
FROM cust_ltv
GROUP BY 1, 2
ORDER BY 1, 2;

COMMENT ON VIEW kpi_customer_lifetime IS
'CLV analysis by signup-month cohort × initial customer_segment.';


-- ----------------------------------------------------------------------
-- 06) kpi_top_10_customers   — Pareto analysis input
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW kpi_top_10_customers AS
WITH per_cust AS (
    SELECT
        c.customer_id,
        c.first_name || ' ' || c.last_name                                      AS name,
        c.state,
        c.customer_segment,
        COUNT(DISTINCT o.order_id)                                              AS total_orders,
        COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                THEN o.order_total ELSE 0 END), 0)                              AS total_revenue,
        COALESCE(SUM(oi.line_total - oi.quantity * p.cost_price), 0)            AS total_profit
    FROM customers c
    JOIN orders o         ON o.customer_id = c.customer_id
    JOIN order_items oi   ON oi.order_id   = o.order_id
    JOIN products p       ON p.product_id  = oi.product_id
    WHERE o.order_status IN ('Delivered','Shipped','Processing','Returned')
    GROUP BY 1, 2, 3, 4
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY total_profit DESC)                          AS profit_rank,
        100.0 * total_profit / NULLIF(SUM(total_profit) OVER (), 0)             AS profit_share_pct
    FROM per_cust
)
SELECT *
FROM ranked
WHERE profit_rank <= 10
ORDER BY profit_rank;

COMMENT ON VIEW kpi_top_10_customers IS
'Top 10 customers by gross profit (used to confirm Pareto 80/20 / 10/52 split).';


-- ----------------------------------------------------------------------
-- 07) kpi_payment_methods
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW kpi_payment_methods AS
SELECT
    pay.payment_method,
    pay.payment_status,
    COUNT(*)                                                                   AS transactions,
    ROUND(SUM(pay.amount), 2)                                                  AS amount_inr,
    ROUND(AVG(pay.amount), 2)                                                  AS avg_ticket_inr
FROM payments pay
GROUP BY 1, 2
ORDER BY amount_inr DESC;

COMMENT ON VIEW kpi_payment_methods IS 'Payment mix: method × status, counts, totals, average tickets.';
