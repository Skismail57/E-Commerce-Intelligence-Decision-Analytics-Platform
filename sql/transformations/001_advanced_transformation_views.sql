-- ======================================================================
-- TRANSFORMATION SQL VIEWS
--   • vw_inventory_analytics   — Stock turnover, days-of-inventory, stock-outs
--   • vw_demand_daily         — Daily demand by product with moving averages
--   • vw_anomaly_prep         — Daily KPIs with z-scores and IQR bounds
--   • vw_supplier_performance — Supplier-level delivery and cost KPIs
--   • vw_store_performance    — Store-level revenue and efficiency
-- ======================================================================

SET search_path TO public;


-- ----------------------------------------------------------------------
-- 01) vw_inventory_analytics
--     Stock turnover ratio, days of inventory, stock-out events,
--     overstock value per product×store.
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_inventory_analytics AS
WITH daily_demand AS (
    SELECT
        oi.product_id,
        DATE_TRUNC('day', o.order_date)::DATE                      AS day,
        COALESCE(SUM(oi.quantity), 0)                               AS qty_sold
    FROM order_items oi
    JOIN orders o ON o.order_id = oi.order_id
    WHERE o.order_status IN ('Delivered','Shipped','Processing','Returned')
    GROUP BY 1, 2
),
demand_30d AS (
    SELECT
        product_id,
        AVG(qty_sold)                                               AS avg_daily_demand_30d
    FROM daily_demand
    WHERE day >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY 1
),
demand_90d AS (
    SELECT
        product_id,
        AVG(qty_sold)                                               AS avg_daily_demand_90d,
        STDDEV(qty_sold)                                            AS demand_std_90d
    FROM daily_demand
    WHERE day >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY 1
),
stockout_events AS (
    SELECT
        inv.product_id,
        inv.store_id,
        COUNT(*) FILTER (WHERE inv.stock_quantity = 0)               AS stockout_count,
        COUNT(*)                                                    AS total_checks,
        MIN(inv.last_restock_date)                                  AS first_restock,
        MAX(inv.last_restock_date)                                  AS last_restock
    FROM inventory inv
    GROUP BY 1, 2
)
SELECT
    inv.inventory_id,
    inv.product_id,
    p.product_name,
    cat.category_name,
    cat.subcategory,
    inv.store_id,
    s.store_name,
    s.city                                                        AS store_city,
    s.state                                                       AS store_state,
    inv.stock_quantity,
    inv.reorder_level,
    inv.safety_stock,
    inv.average_daily_demand,
    COALESCE(d30.avg_daily_demand_30d, 0)                         AS daily_demand_30d,
    COALESCE(d90.avg_daily_demand_90d, 0)                         AS daily_demand_90d,
    COALESCE(d90.demand_std_90d, 0)                               AS demand_volatility,
    COALESCE(so.stockout_count, 0)                                AS stockout_count_30d,
    -- Stock health metrics
    CASE WHEN COALESCE(d30.avg_daily_demand_30d, 0) > 0
         THEN ROUND(inv.stock_quantity / d30.avg_daily_demand_30d, 1)
         ELSE NULL END                                            AS days_of_inventory,
    CASE WHEN COALESCE(d90.avg_daily_demand_90d, 0) > 0
         THEN ROUND(90.0 * inv.stock_quantity /
               NULLIF(d90.avg_daily_demand_90d * 90, 0), 2)
         ELSE 0 END                                               AS stock_turnover_90d,
    -- Reorder recommendations
    ROUND(
        COALESCE(d30.avg_daily_demand_30d, 0) * COALESCE(inv.lead_time_days, 7)
      + COALESCE(inv.safety_stock, 0), 0)                         AS recommended_reorder_point,
    GREATEST(0, ROUND(
        COALESCE(d30.avg_daily_demand_30d, 0) * COALESCE(inv.lead_time_days, 7) * 2
      + COALESCE(inv.safety_stock, 0) - inv.stock_quantity, 0))   AS recommended_reorder_qty,
    -- Safety stock (service level 95% → z = 1.645)
    ROUND(1.645 * COALESCE(d90.demand_std_90d, 0)
        * SQRT(COALESCE(inv.lead_time_days, 7)), 0)              AS safety_stock_calc,
    -- Inventory value
    inv.stock_quantity * p.cost_price                             AS inventory_value_cost,
    inv.stock_quantity * p.selling_price                          AS inventory_value_retail,
    ROUND(100.0 * NULLIF(inv.stock_quantity * p.cost_price, 0)
                / NULLIF(inv.stock_quantity * p.selling_price, 0), 2) AS cost_to_retail_pct,
    -- Overstock detection: > 5x reorder_level AND > 90 days of stock
    CASE WHEN inv.stock_quantity > COALESCE(inv.reorder_level, 50) * 5
              AND (CASE WHEN COALESCE(d30.avg_daily_demand_30d, 0) > 0
                        THEN inv.stock_quantity / d30.avg_daily_demand_30d
                        ELSE 0 END) > 90
         THEN 'Overstock'
         WHEN inv.stock_quantity = 0 THEN 'Out of Stock'
         WHEN inv.stock_quantity <= COALESCE(inv.safety_stock, 0) THEN 'Critical'
         WHEN inv.stock_quantity <= COALESCE(inv.reorder_level, 50) THEN 'Reorder'
         ELSE 'Healthy' END                                       AS inventory_status
FROM inventory inv
JOIN products p   ON p.product_id = inv.product_id
JOIN categories cat ON cat.category_id = p.category_id
JOIN stores s     ON s.store_id   = inv.store_id
LEFT JOIN demand_30d d30 ON d30.product_id = inv.product_id
LEFT JOIN demand_90d d90 ON d90.product_id = inv.product_id
LEFT JOIN stockout_events so
    ON so.product_id = inv.product_id AND so.store_id = inv.store_id;

COMMENT ON VIEW vw_inventory_analytics IS
'Inventory analytics: turnover, days-of-stock, stock-outs, reorder recommendations, overstock flag.';


-- ----------------------------------------------------------------------
-- 02) vw_demand_daily
--     Daily demand series by product×category with calendar features
--     and rolling averages for forecasting model input.
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_demand_daily AS
WITH order_days AS (
    SELECT
        DATE_TRUNC('day', o.order_date)::DATE                      AS day,
        oi.product_id,
        SUM(oi.quantity)                                           AS units_sold,
        SUM(oi.line_total)                                         AS revenue_inr
    FROM order_items oi
    JOIN orders o ON o.order_id = oi.order_id
    WHERE o.order_status IN ('Delivered','Shipped','Processing','Returned')
    GROUP BY 1, 2
),
date_range AS (
    SELECT generate_series(
        COALESCE((SELECT MIN(day) FROM order_days), CURRENT_DATE - INTERVAL '365 days'),
        COALESCE((SELECT MAX(day) FROM order_days), CURRENT_DATE),
        '1 day'::INTERVAL
    )::DATE AS day
),
product_list AS (
    SELECT DISTINCT product_id, product_name, category_id, selling_price
    FROM products
)
SELECT
    d.day,
    EXTRACT(isodow FROM d.day)                                    AS dow,
    EXTRACT(day FROM d.day)                                       AS dom,
    EXTRACT(week FROM d.day)                                      AS iso_week,
    EXTRACT(month FROM d.day)                                     AS month,
    EXTRACT(quarter FROM d.day)                                   AS quarter,
    EXTRACT(year FROM d.day)                                      AS year,
    CASE WHEN EXTRACT(isodow FROM d.day) IN (6,7) THEN 1 ELSE 0 END AS is_weekend,
    pl.product_id,
    pl.product_name,
    cat.category_name,
    cat.subcategory,
    pl.selling_price,
    COALESCE(od.units_sold, 0)                                    AS units_sold,
    COALESCE(od.revenue_inr, 0)                                   AS revenue_inr,
    -- 7-day moving average
    ROUND(AVG(COALESCE(od.units_sold, 0)) OVER (
        PARTITION BY pl.product_id
        ORDER BY d.day
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2)                                                         AS units_7d_ma,
    -- 28-day moving average (4 weeks)
    ROUND(AVG(COALESCE(od.units_sold, 0)) OVER (
        PARTITION BY pl.product_id
        ORDER BY d.day
        ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
    ), 2)                                                         AS units_28d_ma,
    -- Year-over-year same-day comparison
    ROUND(AVG(COALESCE(od.units_sold, 0)) OVER (
        PARTITION BY pl.product_id
        ORDER BY d.day
        ROWS BETWEEN 364 PRECEDING AND 364 PRECEDING
    ), 2)                                                         AS units_yoy_same_day,
    -- Active marketing campaigns on this day
    COUNT(DISTINCT mc.campaign_id)                                AS active_campaigns,
    -- Festival/season flags (Indian context)
    CASE
        WHEN (EXTRACT(month FROM d.day) = 10 AND EXTRACT(day FROM d.day) BETWEEN 15 AND 25)
          OR (EXTRACT(month FROM d.day) = 11 AND EXTRACT(day FROM d.day) BETWEEN 1 AND 10)
        THEN 1 ELSE 0 END                                         AS is_diwali_season,
    CASE WHEN EXTRACT(month FROM d.day) = 1  AND EXTRACT(day FROM d.day) BETWEEN 20 AND 31 THEN 1 ELSE 0 END AS is_republic_day_sale,
    CASE WHEN EXTRACT(month FROM d.day) = 8  AND EXTRACT(day FROM d.day) BETWEEN 10 AND 20 THEN 1 ELSE 0 END AS is_independence_sale,
    CASE WHEN EXTRACT(month FROM d.day) = 12 AND EXTRACT(day FROM d.day) BETWEEN 20 AND 31 THEN 1 ELSE 0 END AS is_christmas_sale
FROM date_range d
CROSS JOIN product_list pl
JOIN categories cat ON cat.category_id = pl.category_id
LEFT JOIN order_days od ON od.day = d.day AND od.product_id = pl.product_id
LEFT JOIN marketing_campaigns mc
    ON d.day BETWEEN mc.start_date AND mc.end_date
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
ORDER BY pl.product_id, d.day;

COMMENT ON VIEW vw_demand_daily IS
'Daily demand training set for forecasting: calendar features, MAs, YoY, campaign flags, Indian seasonal flags.';


-- ----------------------------------------------------------------------
-- 03) vw_anomaly_prep
--     Daily aggregated KPIs with z-scores and IQR bounds ready for
--     anomaly detection (statistical or model-based).
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_anomaly_prep AS
WITH daily_kpis AS (
    SELECT
        DATE_TRUNC('day', o.order_date)::DATE                      AS day,
        ROUND(COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                    THEN o.order_total ELSE 0 END), 0), 2)        AS total_revenue_inr,
        COUNT(DISTINCT o.order_id)                                 AS total_orders,
        COUNT(DISTINCT o.customer_id)                              AS unique_customers,
        ROUND(1.0 * NULLIF(
            COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                          THEN o.order_total ELSE 0 END), 0), 0)
          / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)             AS avg_order_value_inr,
        ROUND(100.0 * NULLIF(
            COUNT(DISTINCT CASE WHEN o.order_status = 'Returned' THEN o.order_id END), 0)
          / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)             AS return_rate_pct,
        COALESCE(SUM(oi.quantity), 0)                              AS units_sold
    FROM orders o
    LEFT JOIN order_items oi ON oi.order_id = o.order_id
    GROUP BY 1
),
stats AS (
    SELECT
        COUNT(*)                                                   AS n_days,
        AVG(total_revenue_inr)                                     AS revenue_mean,
        STDDEV(total_revenue_inr)                                  AS revenue_std,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_revenue_inr) AS revenue_q25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_revenue_inr) AS revenue_q75,
        AVG(total_orders)                                          AS orders_mean,
        STDDEV(total_orders)                                       AS orders_std,
        AVG(return_rate_pct)                                       AS return_mean,
        STDDEV(return_rate_pct)                                    AS return_std
    FROM daily_kpis
)
SELECT
    dk.*,
    st.n_days,
    -- Z-scores
    ROUND(NULLIF(dk.total_revenue_inr - st.revenue_mean, 0)
        / NULLIF(st.revenue_std, 0), 3)                            AS revenue_zscore,
    ROUND(NULLIF(dk.total_orders - st.orders_mean, 0)
        / NULLIF(st.orders_std, 0), 3)                             AS orders_zscore,
    ROUND(NULLIF(dk.return_rate_pct - st.return_mean, 0)
        / NULLIF(st.return_std, 0), 3)                             AS return_zscore,
    -- IQR bounds for revenue
    st.revenue_q25,
    st.revenue_q75,
    st.revenue_q25 - 1.5 * (st.revenue_q75 - st.revenue_q25)      AS revenue_iqr_lower,
    st.revenue_q75 + 1.5 * (st.revenue_q75 - st.revenue_q25)      AS revenue_iqr_upper,
    -- Anomaly flags
    CASE WHEN ABS(NULLIF(dk.total_revenue_inr - st.revenue_mean, 0)
              / NULLIF(st.revenue_std, 0)) > 3 THEN 1 ELSE 0 END  AS revenue_anomaly_z3,
    CASE WHEN dk.total_revenue_inr < (st.revenue_q25 - 1.5 * (st.revenue_q75 - st.revenue_q25))
           OR dk.total_revenue_inr > (st.revenue_q75 + 1.5 * (st.revenue_q75 - st.revenue_q25))
         THEN 1 ELSE 0 END                                        AS revenue_anomaly_iqr,
    -- Category flags
    CASE WHEN dk.total_revenue_inr < st.revenue_mean - 2 * st.revenue_std
         THEN '⚠️  Revenue Drop'
         WHEN dk.total_revenue_inr > st.revenue_mean + 2 * st.revenue_std
         THEN '📈 Revenue Spike'
         WHEN dk.return_rate_pct > st.return_mean + 2 * st.return_std
         THEN '🔁 Return Spike'
         ELSE 'Normal' END::VARCHAR(50)                            AS anomaly_label
FROM daily_kpis dk, stats st
ORDER BY dk.day DESC;

COMMENT ON VIEW vw_anomaly_prep IS
'Daily KPIs pre-computed with z-scores, IQR bounds, and anomaly flags for automated anomaly detection.';


-- ----------------------------------------------------------------------
-- 04) vw_supplier_performance
--     Supplier 360: orders, revenue, cost, lead-time, return rate,
--     inventory health per supplier.
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_supplier_performance AS
WITH sup_product_sales AS (
    SELECT
        p.supplier_id,
        COUNT(DISTINCT oi.order_id)                               AS orders_supplied,
        SUM(oi.quantity)                                          AS units_supplied,
        ROUND(SUM(oi.line_total), 2)                              AS revenue_inr,
        ROUND(SUM(oi.quantity * p.cost_price), 2)                 AS total_cogs,
        ROUND(100.0 * NULLIF(SUM(oi.line_total) - SUM(oi.quantity * p.cost_price), 0)
                    / NULLIF(SUM(oi.line_total), 0), 2)          AS avg_margin_pct
    FROM order_items oi
    JOIN orders o   ON o.order_id   = oi.order_id
    JOIN products p ON p.product_id = oi.product_id
    WHERE o.order_status IN ('Delivered','Shipped','Processing','Returned')
    GROUP BY 1
),
sup_returns AS (
    SELECT
        p.supplier_id,
        COUNT(DISTINCT r.return_id)                               AS return_count,
        SUM(r.quantity_returned)                                  AS return_qty,
        ROUND(SUM(r.refund_amount), 2)                            AS refund_value_inr
    FROM returns r
    JOIN products p ON p.product_id = r.product_id
    GROUP BY 1
),
sup_inventory AS (
    SELECT
        p.supplier_id,
        COUNT(DISTINCT inv.inventory_id)                          AS sku_store_count,
        SUM(inv.stock_quantity)                                   AS total_stock_units,
        ROUND(SUM(inv.stock_quantity * p.cost_price), 2)         AS stock_value_cost,
        COUNT(DISTINCT CASE WHEN inv.stock_quantity <= COALESCE(inv.reorder_level, 50)
                            THEN inv.inventory_id END)           AS low_stock_count
    FROM inventory inv
    JOIN products p ON p.product_id = inv.product_id
    GROUP BY 1
),
sup_product_count AS (
    SELECT supplier_id, COUNT(*) AS product_count
    FROM products GROUP BY 1
)
SELECT
    s.supplier_id,
    s.supplier_name,
    s.city,
    s.state,
    s.rating                                                    AS supplier_rating,
    s.reliability_score,
    s.lead_time_days                                            AS contract_lead_time_days,
    COALESCE(pc.product_count, 0)                               AS sku_count,
    COALESCE(sps.orders_supplied, 0)                            AS orders_supplied,
    COALESCE(sps.units_supplied, 0)                             AS units_supplied,
    COALESCE(sps.revenue_inr, 0)                                AS revenue_inr,
    COALESCE(sps.total_cogs, 0)                                 AS total_cogs_inr,
    COALESCE(sps.avg_margin_pct, 0)                             AS avg_margin_pct,
    COALESCE(sr.return_count, 0)                                AS return_count,
    COALESCE(sr.return_qty, 0)                                  AS return_qty,
    ROUND(100.0 * NULLIF(COALESCE(sr.return_qty, 0), 0)
                / NULLIF(COALESCE(sps.units_supplied, 0), 0), 2) AS return_rate_pct,
    COALESCE(si.sku_store_count, 0)                             AS sku_store_count,
    COALESCE(si.total_stock_units, 0)                           AS total_stock_units,
    COALESCE(si.stock_value_cost, 0)                            AS stock_value_cost_inr,
    COALESCE(si.low_stock_count, 0)                             AS low_stock_sku_stores,
    -- Composite supplier score
    ROUND(
        COALESCE(s.rating, 0) * 20
      + COALESCE(s.reliability_score, 0) * 0.4
      - COALESCE(100.0 * NULLIF(COALESCE(sr.return_qty, 0), 0)
                 / NULLIF(COALESCE(sps.units_supplied, 1), 1), 0) * 0.5,
    2)                                                           AS composite_score
FROM suppliers s
LEFT JOIN sup_product_sales sps  ON sps.supplier_id = s.supplier_id
LEFT JOIN sup_returns sr         ON sr.supplier_id  = s.supplier_id
LEFT JOIN sup_inventory si       ON si.supplier_id  = s.supplier_id
LEFT JOIN sup_product_count pc   ON pc.supplier_id  = s.supplier_id
ORDER BY COALESCE(sps.revenue_inr, 0) DESC;

COMMENT ON VIEW vw_supplier_performance IS
'Supplier 360°: sales volumes, margin, return rate, inventory investment, composite performance score.';


-- ----------------------------------------------------------------------
-- 05) vw_store_performance
--     Store 360: revenue, orders, inventory turns, staffing KPIs.
-- ----------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_store_performance AS
WITH store_sales AS (
    SELECT
        o.store_id,
        COUNT(DISTINCT o.order_id)                               AS orders,
        ROUND(COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                    THEN o.order_total ELSE 0 END), 0), 2)      AS revenue_inr,
        COUNT(DISTINCT o.customer_id)                            AS unique_customers,
        ROUND(1.0 * NULLIF(
            COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                          THEN o.order_total ELSE 0 END), 0), 0)
          / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)           AS avg_order_value_inr,
        ROUND(100.0 * NULLIF(
            COUNT(DISTINCT CASE WHEN o.order_status = 'Returned' THEN o.order_id END), 0)
          / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)           AS return_rate_pct,
        COALESCE(SUM(CASE WHEN o.order_status IN ('Delivered','Shipped','Processing','Returned')
                    THEN o.shipping_cost ELSE 0 END), 0)        AS shipping_cost_inr
    FROM orders o
    GROUP BY 1
),
store_inventory AS (
    SELECT
        store_id,
        COUNT(DISTINCT product_id)                               AS skus_carried,
        SUM(stock_quantity)                                      AS total_stock_units,
        ROUND(SUM(stock_quantity * COALESCE(p.cost_price, 0)), 2)   AS stock_value_cost,
        ROUND(SUM(stock_quantity * COALESCE(p.selling_price, 0)), 2) AS stock_value_retail,
        COUNT(DISTINCT CASE WHEN stock_quantity <= COALESCE(reorder_level, 50)
                            THEN inventory_id END)               AS low_stock_count
    FROM inventory inv
    LEFT JOIN products p ON p.product_id = inv.product_id
    GROUP BY 1
),
store_staff AS (
    SELECT
        store_id,
        COUNT(*)                                                 AS headcount,
        AVG(performance_score)                                   AS avg_staff_performance
    FROM employees
    WHERE store_id IS NOT NULL
    GROUP BY 1
)
SELECT
    s.store_id,
    s.store_name,
    s.store_type,
    s.city,
    s.state,
    s.opening_date,
    COALESCE(ss.orders, 0)                                       AS orders,
    COALESCE(ss.revenue_inr, 0)                                  AS revenue_inr,
    COALESCE(ss.unique_customers, 0)                             AS unique_customers,
    COALESCE(ss.avg_order_value_inr, 0)                          AS aov_inr,
    COALESCE(ss.return_rate_pct, 0)                              AS return_rate_pct,
    COALESCE(ss.shipping_cost_inr, 0)                            AS shipping_cost_inr,
    COALESCE(si.skus_carried, 0)                                 AS skus_carried,
    COALESCE(si.total_stock_units, 0)                            AS total_stock_units,
    COALESCE(si.stock_value_cost, 0)                             AS stock_value_cost_inr,
    COALESCE(si.stock_value_retail, 0)                           AS stock_value_retail_inr,
    COALESCE(si.low_stock_count, 0)                              AS low_stock_skus,
    COALESCE(st.headcount, 0)                                    AS headcount,
    ROUND(COALESCE(st.avg_staff_performance, 0), 2)              AS avg_staff_performance,
    -- Derived KPIs
    ROUND(1.0 * NULLIF(COALESCE(ss.revenue_inr, 0), 0)
                / NULLIF(COALESCE(st.headcount, 1), 1), 2)      AS revenue_per_employee,
    ROUND(1.0 * NULLIF(COALESCE(ss.revenue_inr, 0), 0)
                / NULLIF(COALESCE(si.stock_value_cost, 1), 1), 2) AS inventory_turns_ratio,
    ROUND(100.0 * NULLIF(COALESCE(ss.shipping_cost_inr, 0), 0)
                / NULLIF(COALESCE(ss.revenue_inr, 1), 1), 2)   AS shipping_pct_of_revenue
FROM stores s
LEFT JOIN store_sales ss      ON ss.store_id = s.store_id
LEFT JOIN store_inventory si  ON si.store_id = s.store_id
LEFT JOIN store_staff st      ON st.store_id = s.store_id
ORDER BY COALESCE(ss.revenue_inr, 0) DESC;

COMMENT ON VIEW vw_store_performance IS
'Store 360°: revenue, orders, inventory turns, staffing efficiency, return rates.';
