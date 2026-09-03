-- ======================================================================
-- STAGING VIEWS
-- Denormalised / convenience views built on top of raw tables
-- These are the "source of truth" views that Power BI and downstream
-- analytics (Python, Streamlit, FastAPI) should consume.
-- ======================================================================

SET search_path TO public;

-- Set analysis as-of date (should match DATA_END_DATE)
SET LOCAL app.analysis_as_of_date = '2024-12-31'::DATE;

-- ----------------------------------------------------------------------
-- 01) vw_customer_360   — Single view per customer: KPIs, RFM scores,
--                         lifetime stats, signup/device info.
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_customer_360 AS
WITH customer_order_metrics AS (
    SELECT
        customer_id,
        COUNT(DISTINCT order_id)                                          AS total_orders,
        COUNT(DISTINCT CASE WHEN order_status = 'Cancelled' THEN order_id END) AS cancelled_orders,
        COUNT(DISTINCT CASE WHEN order_status = 'Returned'  THEN order_id END) AS returned_orders,
        COALESCE(SUM(CASE WHEN order_status IN ('Delivered','Shipped','Processing','Returned')
                    THEN order_total ELSE 0 END), 0)                       AS total_revenue,
        COALESCE(SUM(CASE WHEN order_status IN ('Delivered','Shipped','Processing','Returned')
                    THEN discount_amount ELSE 0 END), 0)                   AS total_discount,
        ROUND(AVG(CASE WHEN order_status IN ('Delivered','Shipped','Processing','Returned')
                    THEN order_total END), 2)                              AS avg_order_value,
        MIN(order_date)::DATE                                              AS first_order_date,
        MAX(order_date)::DATE                                              AS last_order_date,
        COUNT(DISTINCT DATE_TRUNC('month', order_date))                    AS active_months
    FROM orders
    GROUP BY customer_id
),
customer_product_metrics AS (
    SELECT
        o.customer_id,
        COUNT(DISTINCT oi.product_id)                                        AS unique_products_bought
    FROM orders o
    INNER JOIN order_items oi ON oi.order_id = o.order_id
    GROUP BY o.customer_id
),
customer_return_metrics AS (
    SELECT
        customer_id,
        COUNT(return_id)                                                   AS total_returns
    FROM returns
    GROUP BY customer_id
),
customer_review_metrics AS (
    SELECT
        customer_id,
        COALESCE(AVG(rating), 0)                                            AS avg_rating_given,
        COUNT(review_id)                                                    AS reviews_written
    FROM reviews
    GROUP BY customer_id
),
customer_payment_metrics AS (
    SELECT
        customer_id,
        COALESCE(SUM(amount) FILTER (WHERE payment_status = 'Success'), 0) AS total_paid
    FROM payments
    GROUP BY customer_id
),
customer_session_metrics AS (
    SELECT
        customer_id,
        COUNT(DISTINCT session_id)                                         AS website_sessions
    FROM website_sessions
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    c.gender,
    c.age,
    c.city,
    c.state,
    c.country,
    c.signup_date,
    c.customer_segment,
    c.signup_channel,
    COALESCE(o_metrics.total_orders, 0)                                   AS total_orders,
    COALESCE(o_metrics.cancelled_orders, 0)                               AS cancelled_orders,
    COALESCE(o_metrics.returned_orders, 0)                                AS returned_orders,
    COALESCE(o_metrics.total_revenue, 0)                                  AS total_revenue,
    COALESCE(o_metrics.total_discount, 0)                                  AS total_discount,
    COALESCE(o_metrics.avg_order_value, 0)                                AS avg_order_value,
    o_metrics.first_order_date                                            AS first_order_date,
    o_metrics.last_order_date                                             AS last_order_date,
    CASE WHEN o_metrics.last_order_date IS NOT NULL
         THEN '2024-12-31'::DATE - o_metrics.last_order_date
         ELSE '2024-12-31'::DATE - c.signup_date END                       AS days_since_last_order,
    COALESCE(o_metrics.active_months, 0)                                  AS active_months,
    COALESCE(p_metrics.unique_products_bought, 0)                         AS unique_products_bought,
    COALESCE(r_metrics.total_returns, 0)                                   AS total_returns,
    COALESCE(rv_metrics.avg_rating_given, 0)                               AS avg_rating_given,
    COALESCE(rv_metrics.reviews_written, 0)                                AS reviews_written,
    COALESCE(pay_metrics.total_paid, 0)                                    AS total_paid,
    COALESCE(ws_metrics.website_sessions, 0)                               AS website_sessions
FROM customers c
LEFT JOIN customer_order_metrics o_metrics       ON o_metrics.customer_id = c.customer_id
LEFT JOIN customer_product_metrics p_metrics     ON p_metrics.customer_id = c.customer_id
LEFT JOIN customer_return_metrics r_metrics      ON r_metrics.customer_id = c.customer_id
LEFT JOIN customer_review_metrics rv_metrics     ON rv_metrics.customer_id = c.customer_id
LEFT JOIN customer_payment_metrics pay_metrics   ON pay_metrics.customer_id = c.customer_id
LEFT JOIN customer_session_metrics ws_metrics    ON ws_metrics.customer_id = c.customer_id;

COMMENT ON VIEW vw_customer_360 IS
'Per-customer 360° view — lifetime revenue, orders, frequency, returns, reviews, sessions (aggregated independently to prevent fan-out).';


-- ----------------------------------------------------------------------
-- 02) vw_product_360   — Per product: revenue, units, margin, returns,
--                        ratings, inventory health.
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_product_360 AS
SELECT
    p.product_id,
    p.product_name,
    p.sku_code,
    p.brand_name,
    p.cost_price,
    p.selling_price,
    ROUND(CASE WHEN p.selling_price > 0
          THEN (p.selling_price - p.cost_price) / p.selling_price * 100 ELSE 0 END, 2)
                                                                         AS product_margin_pct,
    cat.category_name,
    cat.subcategory,
    sup.supplier_name,
    sup.rating                                                            AS supplier_rating,
    p.launch_date,
    p.product_status,
    COUNT(DISTINCT oi.order_id) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing'))
                                                                         AS total_orders,
    COALESCE(SUM(oi.quantity) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0)
                                                                         AS units_sold,
    COALESCE(SUM(oi.line_total) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0)
                                                                         AS revenue,
    COALESCE(SUM(oi.quantity * p.cost_price) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0)
                                                                         AS cogs,
    COALESCE(SUM(oi.line_total) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0)
    - COALESCE(SUM(oi.quantity * p.cost_price) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0)
                                                                         AS gross_profit,
    ROUND(
        (
            COALESCE(SUM(oi.line_total) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0)
          - COALESCE(SUM(oi.quantity * p.cost_price) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0)
        )::NUMERIC / NULLIF(
            COALESCE(SUM(oi.line_total) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0), 0) * 100, 2)
                                                                         AS gross_margin_pct,
    COALESCE(SUM(oi.discount) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0)
                                                                         AS discount_given,
    COALESCE(AVG(oi.discount_pct) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0)
                                                                         AS avg_discount_pct,
    COUNT(DISTINCT r.return_id)                                          AS total_returns,
    ROUND(
        CASE WHEN COALESCE(SUM(oi.quantity) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0) > 0
        THEN COUNT(DISTINCT r.return_id) * 100.0 /
             NULLIF(COALESCE(SUM(oi.quantity) FILTER (WHERE o.order_status IN ('Delivered','Returned','Shipped','Processing')), 0), 0)
        ELSE 0 END, 2)                                                   AS return_rate_pct,
    COALESCE(AVG(rv.rating), 0)                                          AS avg_rating,
    COUNT(rv.review_id)                                                  AS total_reviews,
    COALESCE(SUM(CASE WHEN rv.rating = 5 THEN 1 ELSE 0 END), 0)          AS five_star_reviews,
    SUM(CASE WHEN inv.stock_quantity IS NOT NULL THEN inv.stock_quantity ELSE 0 END)
                                                                         AS current_stock_units,
    COALESCE(SUM(CASE WHEN inv.stock_quantity IS NOT NULL
                THEN inv.stock_quantity * p.cost_price ELSE 0 END), 0)   AS inventory_value_cost,
    COALESCE(SUM(CASE WHEN inv.stock_quantity IS NOT NULL
                THEN inv.stock_quantity * p.selling_price ELSE 0 END), 0) AS inventory_value_retail,
    COUNT(DISTINCT CASE WHEN inv.stock_quantity <= COALESCE(inv.reorder_level,50)
                   THEN inv.inventory_id END)                            AS at_risk_stores_count
FROM products p
LEFT JOIN categories cat            ON cat.category_id  = p.category_id
LEFT JOIN suppliers  sup            ON sup.supplier_id  = p.supplier_id
LEFT JOIN order_items oi            ON oi.product_id    = p.product_id
LEFT JOIN orders o                  ON o.order_id       = oi.order_id
LEFT JOIN returns r                 ON r.product_id     = p.product_id
LEFT JOIN reviews rv                ON rv.product_id    = p.product_id
LEFT JOIN inventory inv             ON inv.product_id   = p.product_id
GROUP BY p.product_id, p.product_name, p.sku_code, p.brand_name,
         p.cost_price, p.selling_price, cat.category_name, cat.subcategory,
         sup.supplier_name, sup.rating, p.launch_date, p.product_status;

COMMENT ON VIEW vw_product_360 IS
'Per-product 360° view — revenue, margin, units, returns, reviews, inventory value.';


-- ----------------------------------------------------------------------
-- 03) vw_order_line_fact   — Fully-denormalised order line fact table
--                            (handy for BI tools & dataframes).
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_order_line_fact AS
SELECT
    oi.order_item_id,
    o.order_id,
    o.order_date,
    DATE_TRUNC('day',   o.order_date)::DATE                                   AS order_day,
    DATE_TRUNC('week',  o.order_date)::DATE                                   AS order_week,
    DATE_TRUNC('month', o.order_date)::DATE                                   AS order_month,
    DATE_TRUNC('quarter', o.order_date)::DATE                                 AS order_quarter,
    DATE_TRUNC('year',  o.order_date)::DATE                                   AS order_year,
    EXTRACT(isodow FROM o.order_date)                                         AS order_dow,
    to_char(o.order_date, 'Dy')                                               AS order_dow_name,
    EXTRACT(hour FROM o.order_date)                                           AS order_hour,
    c.customer_id,
    c.first_name || ' ' || c.last_name                                        AS customer_name,
    c.gender,
    c.age,
    c.city                                                                    AS customer_city,
    c.state                                                                   AS customer_state,
    c.customer_segment,
    c.signup_channel,
    p.product_id,
    p.product_name,
    p.sku_code,
    p.brand_name,
    cat.category_name,
    cat.subcategory,
    sup.supplier_name,
    s.store_id,
    s.store_name,
    s.city                                                                    AS store_city,
    s.state                                                                   AS store_state,
    mc.campaign_id,
    mc.campaign_name,
    mc.channel                                                                AS campaign_channel,
    pay.payment_id,
    pay.payment_method,
    pay.payment_status,
    o.order_status,
    o.device_type,
    oi.quantity,
    oi.unit_price,
    oi.discount,
    oi.discount_pct,
    oi.tax,
    oi.line_total                                                             AS line_revenue,
    p.cost_price                                                              AS unit_cost,
    (oi.quantity * p.cost_price)                                              AS line_cogs,
    (oi.line_total - (oi.quantity * p.cost_price))                            AS line_gross_profit,
    ROUND(
        CASE WHEN oi.line_total > 0
        THEN (oi.line_total - (oi.quantity * p.cost_price)) / oi.line_total * 100
        ELSE 0 END, 2)                                                        AS line_margin_pct,
    o.shipping_cost                                                           AS order_shipping_cost,
    o.discount_amount                                                         AS order_discount,
    o.tax_amount                                                              AS order_tax,
    o.order_total                                                             AS order_total_inc_shipping,
    o.shipping_date,
    o.delivery_date,
    (o.delivery_date::DATE - o.shipping_date::DATE)                           AS shipping_to_delivery_days
FROM order_items oi
JOIN orders o                     ON o.order_id     = oi.order_id
JOIN customers c                  ON c.customer_id  = o.customer_id
JOIN products p                   ON p.product_id   = oi.product_id
JOIN categories cat               ON cat.category_id = p.category_id
JOIN suppliers sup                ON sup.supplier_id = p.supplier_id
LEFT JOIN stores s                ON s.store_id     = o.store_id
LEFT JOIN marketing_campaigns mc  ON mc.campaign_id = o.campaign_id
LEFT JOIN payments pay            ON pay.payment_id = o.payment_id;

COMMENT ON VIEW vw_order_line_fact IS
'Denormalised order line fact table for BI and Pandas analysis.';


-- ----------------------------------------------------------------------
-- 04) vw_daily_sales   — Daily aggregated sales fact
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_daily_sales AS
SELECT
    DATE_TRUNC('day', o.order_date)::DATE                                   AS order_day,
    COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                THEN o.order_total ELSE 0 END), 0)                          AS gross_revenue,
    COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                THEN o.discount_amount ELSE 0 END), 0)                      AS total_discount,
    COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                THEN o.tax_amount ELSE 0 END), 0)                           AS total_tax,
    COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                THEN o.shipping_cost ELSE 0 END), 0)                        AS total_shipping,
    COALESCE(SUM(CASE WHEN o.order_status = 'Returned'
                THEN o.order_total ELSE 0 END), 0)                          AS returned_revenue,
    COALESCE(SUM(CASE WHEN o.order_status = 'Cancelled'
                THEN o.order_total ELSE 0 END), 0)                          AS cancelled_revenue,
    (COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing')
                THEN o.order_total ELSE 0 END), 0)
    - COALESCE(SUM(CASE WHEN o.order_status = 'Returned'
                THEN o.order_total ELSE 0 END), 0))                         AS net_revenue,
    COUNT(DISTINCT o.order_id)                                               AS total_orders,
    COUNT(DISTINCT CASE WHEN o.order_status = 'Delivered'  THEN o.order_id END) AS delivered_orders,
    COUNT(DISTINCT CASE WHEN o.order_status = 'Shipped'    THEN o.order_id END) AS shipped_orders,
    COUNT(DISTINCT CASE WHEN o.order_status = 'Processing' THEN o.order_id END) AS processing_orders,
    COUNT(DISTINCT CASE WHEN o.order_status = 'Returned'   THEN o.order_id END) AS returned_orders,
    COUNT(DISTINCT CASE WHEN o.order_status = 'Cancelled'  THEN o.order_id END) AS cancelled_orders,
    COUNT(DISTINCT o.customer_id)                                            AS unique_customers,
    COUNT(DISTINCT CASE WHEN c2.first_order_date::DATE = DATE_TRUNC('day', o.order_date)::DATE
                   THEN o.customer_id END)                                   AS new_customers,
    COALESCE(SUM(oi.quantity), 0)                                            AS units_sold,
    COUNT(DISTINCT oi.product_id)                                            AS unique_products_sold
FROM orders o
LEFT JOIN order_items oi           ON oi.order_id    = o.order_id
LEFT JOIN (
    SELECT customer_id, MIN(order_date) AS first_order_date FROM orders GROUP BY 1
) c2 ON c2.customer_id = o.customer_id
GROUP BY 1;

COMMENT ON VIEW vw_daily_sales IS 'Daily aggregated sales fact table.';


-- ----------------------------------------------------------------------
-- 05) vw_marketing_channel_funnel
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_marketing_channel_funnel AS
SELECT
    COALESCE(mc.channel, ws.channel, 'Unknown')                              AS channel,
    DATE_TRUNC('day', ws.session_date)::DATE                                 AS day,
    COUNT(DISTINCT ws.session_id)                                            AS sessions,
    SUM(ws.page_views)                                                       AS page_views,
    SUM(ws.product_views)                                                    AS product_views,
    SUM(ws.cart_adds)                                                        AS cart_adds,
    SUM(ws.checkout_started)                                                 AS checkouts_started,
    SUM(ws.checkout_completed)                                               AS checkouts_completed,
    COUNT(DISTINCT CASE WHEN ws.customer_id IS NOT NULL
                   THEN ws.customer_id END)                                  AS signed_in_users,
    ROUND(100.0 * NULLIF(SUM(ws.checkout_completed), 0)
                / NULLIF(COUNT(DISTINCT ws.session_id), 0), 2)               AS conversion_rate_pct,
    COALESCE(SUM(ms.impressions), 0)                                         AS impressions,
    COALESCE(SUM(ms.clicks), 0)                                              AS clicks,
    COALESCE(SUM(ms.spend_amount), 0)                                        AS spend,
    ROUND(100.0 * NULLIF(SUM(ms.clicks), 0)
                / NULLIF(SUM(ms.impressions), 0), 4)                        AS ctr_pct,
    ROUND(1.0 * NULLIF(SUM(ms.spend_amount), 0)
                / NULLIF(SUM(ms.clicks), 0), 4)                             AS cpc,
    ROUND(1.0 * NULLIF(SUM(ms.spend_amount), 0)
                / NULLIF(COUNT(DISTINCT CASE WHEN ws.customer_id IS NOT NULL AND ws.checkout_completed > 0
                                     THEN ws.customer_id END), 0), 0)       AS cac,
    ROUND(1.0 * NULLIF(
        SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
            THEN o.order_total ELSE 0 END), 0)
                / NULLIF(SUM(ms.spend_amount), 0), 2)                       AS roas
FROM website_sessions ws
LEFT JOIN marketing_campaigns mc    ON mc.campaign_id = ws.campaign_id
LEFT JOIN marketing_spend ms        ON ms.channel = COALESCE(mc.channel, ws.channel)
                                     AND ms.spend_date = ws.session_date
LEFT JOIN orders o                  ON o.customer_id = ws.customer_id
                                     AND DATE_TRUNC('day', o.order_date) = ws.session_date
GROUP BY 1, 2;

COMMENT ON VIEW vw_marketing_channel_funnel IS
'Marketing funnel per channel per day: impressions → clicks → sessions → cart → checkout → revenue, plus CAC/ROAS.';


-- ----------------------------------------------------------------------
-- 06) vw_inventory_health   — Stock-outs, overstock, reorder projection
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_inventory_health AS
SELECT
    inv.inventory_id,
    inv.product_id,
    p.product_name,
    cat.category_name,
    inv.store_id,
    s.store_name,
    s.city                                                                    AS store_city,
    inv.stock_quantity,
    inv.reorder_level,
    inv.safety_stock,
    inv.average_daily_demand,
    inv.lead_time_days,
    COALESCE(inv.average_daily_demand, 0) * COALESCE(inv.lead_time_days, 7)  AS lead_time_demand,
    COALESCE(inv.average_daily_demand, 0) * COALESCE(inv.lead_time_days, 7)
  + COALESCE(inv.safety_stock, 0)                                            AS recommended_reorder_point,
    GREATEST(0,
        COALESCE(inv.average_daily_demand, 0) * COALESCE(inv.lead_time_days, 7) * 2
      + COALESCE(inv.safety_stock, 0) - inv.stock_quantity)                  AS recommended_reorder_qty,
    CASE WHEN inv.stock_quantity = 0 THEN 'Out of Stock'
         WHEN inv.stock_quantity <= COALESCE(inv.safety_stock, 0) THEN 'Critical'
         WHEN inv.stock_quantity <= COALESCE(inv.reorder_level, 50) THEN 'Reorder'
         ELSE 'OK' END                                                       AS stock_status,
    CASE WHEN COALESCE(inv.average_daily_demand, 0) > 0
         THEN ROUND(inv.stock_quantity / NULLIF(inv.average_daily_demand, 0), 1)
         ELSE NULL END                                                       AS days_of_inventory,
    inv.last_restock_date,
    inv.stock_quantity * p.cost_price                                        AS inventory_value_cost,
    inv.stock_quantity * p.selling_price                                     AS inventory_value_retail,
    CASE WHEN inv.stock_quantity > COALESCE(inv.reorder_level, 50) * 5
         THEN 'Overstock' ELSE 'Normal' END                                  AS overstock_flag
FROM inventory inv
JOIN products p       ON p.product_id = inv.product_id
JOIN categories cat   ON cat.category_id = p.category_id
JOIN stores s         ON s.store_id   = inv.store_id;

COMMENT ON VIEW vw_inventory_health IS
'Inventory health per product×store: stock status, reorder point, days-of-inventory, overstock flag.';


-- ----------------------------------------------------------------------
-- 07) vw_rfm_segments   — RFM calculation and 7-segment classification
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_rfm_segments AS
WITH customer_base AS (
    SELECT
        c.customer_id,
        c.customer_segment                                                       AS assigned_segment,
        COALESCE(MAX(o.order_date)::DATE, c.signup_date)                       AS last_order_date,
        ('2024-12-31'::DATE - COALESCE(MAX(o.order_date)::DATE, c.signup_date)) AS recency_days,
        COUNT(DISTINCT o.order_id)                                              AS frequency,
        COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                    THEN o.order_total ELSE 0 END), 0)                         AS monetary_value
    FROM customers c
    LEFT JOIN orders o ON o.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_segment, c.signup_date
),
rfm_scores AS (
    SELECT
        *,
        NTILE(5) OVER (ORDER BY recency_days DESC)                              AS r_score,
        CASE WHEN frequency = 0 THEN 1
             ELSE LEAST(NTILE(5) OVER (ORDER BY frequency ASC), 5) END          AS f_score,
        CASE WHEN monetary_value = 0 THEN 1
             ELSE LEAST(NTILE(5) OVER (ORDER BY monetary_value ASC), 5) END     AS m_score
    FROM customer_base
),
rfm_combined AS (
    SELECT
        *,
        (r_score + f_score + m_score)                                           AS rfm_total,
        (r_score * 100 + f_score * 10 + m_score)                               AS rfm_cell
    FROM rfm_scores
)
SELECT
    customer_id,
    assigned_segment,
    last_order_date,
    recency_days,
    frequency,
    monetary_value,
    r_score,
    f_score,
    m_score,
    rfm_total,
    rfm_cell,
    CASE
        WHEN r_score = 5 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 4 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 3 AND f_score >= 1 AND m_score <= 3 AND m_score >= 2 THEN 'Potential Loyalists'
        WHEN r_score = 5 AND f_score <= 2 AND m_score <= 2 THEN 'New Customers'
        WHEN r_score = 2 AND f_score >= 2 AND m_score >= 2 THEN 'At Risk'
        WHEN r_score = 1 AND f_score >= 4 AND m_score >= 4 THEN 'Can''t Lose Them'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN 'Lost Customers'
        ELSE 'Other'
    END                                                                        AS rfm_segment
FROM rfm_combined;

COMMENT ON VIEW vw_rfm_segments IS
'Recency-Frequency-Monetary scores plus 7-way RFM segment classification per customer.';
