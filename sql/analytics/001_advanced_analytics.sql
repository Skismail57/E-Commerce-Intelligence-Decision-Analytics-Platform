-- ======================================================================
-- ANALYTICS SQL: Advanced Analytics Views
--   • Cohort Retention Matrix
--   • Product Matrix (Stars / Volume / Dogs / Premium quadrants)
--   • Churn Feature Prep (for ML model input)
--   • Demand Forecast Training Set (daily × product)
-- ======================================================================

SET search_path TO public;

-- Analysis as-of date - should match dataset end date
-- This prevents using CURRENT_DATE which would be incorrect for historical data
DO $$
BEGIN
    PERFORM set_config('app.analysis_as_of_date', '2024-12-31', false);
END $$;

-- ----------------------------------------------------------------------
-- 01) vw_analytics_cohort_retention
--     N-month retention for each signup-month cohort
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_analytics_cohort_retention AS
WITH signup_cohort AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', signup_date)::DATE                                AS cohort_month
    FROM customers
),
activity AS (
    SELECT
        s.customer_id,
        s.cohort_month,
        DATE_TRUNC('month', o.order_date)::DATE                               AS activity_month,
        EXTRACT(year FROM age(DATE_TRUNC('month', o.order_date), s.cohort_month)) * 12
      + EXTRACT(month FROM age(DATE_TRUNC('month', o.order_date), s.cohort_month))
                                                                              AS month_number
    FROM signup_cohort s
    JOIN orders o ON o.customer_id = s.customer_id
    WHERE o.order_status IN ('Delivered','Shipped','Processing','Returned')
),
per_cohort_size AS (
    SELECT cohort_month, COUNT(DISTINCT customer_id) AS cohort_size FROM signup_cohort GROUP BY 1
)
SELECT
    a.cohort_month,
    cs.cohort_size,
    a.month_number                                                           AS months_since_signup,
    COUNT(DISTINCT a.customer_id)                                            AS active_customers,
    ROUND(100.0 * NULLIF(COUNT(DISTINCT a.customer_id), 0)
                / NULLIF(cs.cohort_size, 0), 2)                             AS retention_pct
FROM activity a
JOIN per_cohort_size cs ON cs.cohort_month = a.cohort_month
WHERE a.month_number BETWEEN 0 AND 11
GROUP BY 1, 2, 3
ORDER BY 1, 3;

COMMENT ON VIEW vw_analytics_cohort_retention IS
'Signup-month cohort × months-since-signup retention matrix (0–11 months).';


-- ----------------------------------------------------------------------
-- 02) vw_analytics_product_matrix
--     Boston-style quadrant: Sales volume × Profit per product
--     Top-right quadrant = "Stars", bottom-right = "Cash Cows / Volume",
--     Top-left = "Dogs / Remove", bottom-left = "Premium / Niche"
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_analytics_product_matrix AS
WITH product_stats AS (
    SELECT
        product_id,
        SUM(units_sold)                                                       AS units_sold,
        SUM(revenue)                                                          AS revenue_inr,
        SUM(gross_profit)                                                     AS profit_inr,
        CASE WHEN SUM(revenue) > 0
             THEN SUM(gross_profit) / SUM(revenue) ELSE 0 END                AS margin_ratio
    FROM vw_product_360
    GROUP BY 1
),
medians AS (
    SELECT
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY revenue_inr)             AS median_revenue,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY margin_ratio)            AS median_margin
    FROM product_stats
)
SELECT
    ps.product_id,
    p360.product_name,
    p360.category_name,
    p360.subcategory,
    p360.brand_name,
    ps.units_sold,
    ROUND(ps.revenue_inr, 2)                                                 AS revenue_inr,
    ROUND(ps.profit_inr, 2)                                                  AS profit_inr,
    ROUND(100.0 * ps.margin_ratio, 2)                                        AS margin_pct,
    m.median_revenue,
    ROUND(100.0 * m.median_margin, 2)                                        AS median_margin_pct,
    CASE
        WHEN ps.revenue_inr  >= m.median_revenue AND ps.margin_ratio >= m.median_margin THEN '⭐ Stars'
        WHEN ps.revenue_inr  >= m.median_revenue AND ps.margin_ratio <  m.median_margin THEN '📦 Volume'
        WHEN ps.revenue_inr  <  m.median_revenue AND ps.margin_ratio <  m.median_margin THEN '❌ Remove'
        ELSE '💎 Premium'
    END                                                                      AS quadrant
FROM product_stats ps
CROSS JOIN medians m
JOIN vw_product_360 p360 ON p360.product_id = ps.product_id;

COMMENT ON VIEW vw_analytics_product_matrix IS
'4-quadrant product matrix: Stars / Volume / Remove / Premium using median revenue × margin cutoffs.';


-- ----------------------------------------------------------------------
-- 03) vw_ml_churn_features
--     One row per customer, input features for churn classification model.
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_ml_churn_features AS
WITH customer_orders AS (
    SELECT
        o.customer_id,
        MIN(o.order_date)::DATE                                               AS first_order_date,
        MAX(o.order_date)::DATE                                               AS last_order_date,
        COUNT(DISTINCT o.order_id)                                            AS total_orders,
        AVG(o.order_total)                                                    AS avg_order_value,
        SUM(o.order_total)                                                    AS total_spend,
        AVG(o.discount_amount)                                                AS avg_discount_amount,
        SUM(CASE WHEN o.order_status = 'Returned' THEN 1 ELSE 0 END)::FLOAT
      / NULLIF(COUNT(DISTINCT o.order_id), 0)                                 AS return_rate
    FROM orders o
    GROUP BY 1
),
customer_sessions AS (
    SELECT
        customer_id,
        COUNT(*)                                                              AS total_sessions,
        AVG(page_views)                                                       AS avg_page_views,
        AVG(session_duration_sec)                                             AS avg_session_sec,
        AVG(checkout_completed)                                               AS checkout_rate
    FROM website_sessions
    WHERE customer_id IS NOT NULL
    GROUP BY 1
),
reviews AS (
    SELECT customer_id, AVG(rating) AS avg_review_rating FROM reviews GROUP BY 1
)
SELECT
    c.customer_id,
    c.age,
    c.gender,
    c.state,
    c.customer_segment,
    (current_setting('app.analysis_as_of_date', true)::DATE - COALESCE(co.last_order_date, c.signup_date)) AS days_since_last_order,
    COALESCE(co.total_orders, 0)::INT                                        AS total_orders,
    ROUND(COALESCE(co.avg_order_value, 0), 2)                                AS avg_order_value,
    ROUND(COALESCE(co.total_spend, 0), 2)                                    AS total_spend,
    ROUND(COALESCE(co.return_rate, 0), 4)                                    AS return_rate,
    ROUND(COALESCE(co.avg_discount_amount, 0), 2)                            AS avg_discount_used,
    COALESCE(s.total_sessions, 0)::INT                                       AS website_sessions,
    ROUND(COALESCE(s.avg_page_views, 0), 2)                                  AS avg_page_views,
    ROUND(COALESCE(s.avg_session_sec, 0), 2)                                 AS avg_session_sec,
    ROUND(COALESCE(r.avg_review_rating, 0), 2)                               AS avg_review_rating,
    (current_setting('app.analysis_as_of_date', true)::DATE - c.signup_date) AS customer_tenure_days,
    CASE
        WHEN (current_setting('app.analysis_as_of_date', true)::DATE - COALESCE(co.last_order_date, c.signup_date)) > 90
         AND COALESCE(co.total_orders, 0) >= 1 THEN 1 ELSE 0 END            AS churn_label_90d
FROM customers c
LEFT JOIN customer_orders co     ON co.customer_id = c.customer_id
LEFT JOIN customer_sessions s    ON s.customer_id  = c.customer_id
LEFT JOIN reviews r              ON r.customer_id  = c.customer_id;

COMMENT ON VIEW vw_ml_churn_features IS
'Training-ready churn feature table (one row per customer): demographics, order, session, review, tenure features + 90-day churn label.';


-- ----------------------------------------------------------------------
-- 04) vw_ml_demand_forecast_train
--     Daily demand by product for forecasting models (ARIMA / Prophet / XGB).
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_ml_demand_forecast_train AS
WITH daily_order_lines AS (
    SELECT
        DATE_TRUNC('day', o.order_date)::DATE                                AS day,
        oi.product_id,
        SUM(oi.quantity)                                                     AS units_sold,
        SUM(oi.line_total)                                                   AS revenue_inr
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status IN ('Delivered','Shipped','Processing','Returned')
    GROUP BY 1, 2
)
SELECT
    d.day,
    EXTRACT(isodow FROM d.day)                                               AS dow,
    EXTRACT(month FROM d.day)                                                AS month,
    EXTRACT(quarter FROM d.day)                                              AS quarter,
    EXTRACT(year FROM d.day)                                                 AS year,
    CASE WHEN EXTRACT(isodow FROM d.day) IN (6,7) THEN 1 ELSE 0 END         AS is_weekend,
    p.product_id,
    p.product_name,
    cat.category_name,
    cat.subcategory,
    p.selling_price,
    COALESCE(ol.units_sold, 0)                                               AS units_sold,
    COALESCE(ol.revenue_inr, 0)                                              AS revenue_inr,
    AVG(COALESCE(ol.units_sold, 0)) OVER (
        PARTITION BY p.product_id
        ORDER BY d.day
        ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
    )                                                                        AS units_7d_avg,
    AVG(COALESCE(ol.units_sold, 0)) OVER (
        PARTITION BY p.product_id
        ORDER BY d.day
        ROWS BETWEEN 27 PRECEDING AND 1 PRECEDING
    )                                                                        AS units_28d_avg,
    SUM(CASE WHEN c.campaign_id IS NOT NULL THEN 1 ELSE 0 END)               AS active_campaigns
FROM generate_series(
        (SELECT MIN(order_date)::DATE FROM orders),
        (SELECT MAX(order_date)::DATE FROM orders),
        '1 day'::INTERVAL) AS d(day)
CROSS JOIN (SELECT DISTINCT product_id, product_name, category_id, selling_price FROM products) p
JOIN categories cat   ON cat.category_id = p.category_id
LEFT JOIN daily_order_lines ol ON ol.day = d.day AND ol.product_id = p.product_id
LEFT JOIN marketing_campaigns c ON d.day BETWEEN c.start_date AND c.end_date
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18
ORDER BY p.product_id, d.day;

COMMENT ON VIEW vw_ml_demand_forecast_train IS
'Complete daily × product training set with calendar, campaign, 7d/28d moving averages.';
