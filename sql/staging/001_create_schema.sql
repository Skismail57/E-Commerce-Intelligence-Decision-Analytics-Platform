-- ======================================================================
-- E-Commerce Intelligence & Decision Analytics Platform
-- PostgreSQL Data Warehouse Schema
-- ======================================================================

SET search_path TO public;

-- ======================================================================
-- DROP EXISTING TABLES (for clean re-runs)
-- ======================================================================

DROP TABLE IF EXISTS website_sessions CASCADE;
DROP TABLE IF EXISTS marketing_spend CASCADE;
DROP TABLE IF EXISTS marketing_campaigns CASCADE;
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS returns CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS inventory CASCADE;
DROP TABLE IF EXISTS employees CASCADE;
DROP TABLE IF EXISTS stores CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS suppliers CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ======================================================================
-- CUSTOMERS
-- ======================================================================

CREATE TABLE customers (
    customer_id         BIGSERIAL PRIMARY KEY,
    first_name        VARCHAR(50) NOT NULL,
    last_name         VARCHAR(50) NOT NULL,
    gender            CHAR(1) CHECK (gender IN ('M', 'F')),
    date_of_birth     DATE,
    age               INTEGER,
    city              VARCHAR(100) NOT NULL,
    state             VARCHAR(100) NOT NULL,
    country           VARCHAR(50) NOT NULL DEFAULT 'India',
    signup_date       DATE NOT NULL,
    customer_segment  VARCHAR(30),
    signup_channel    VARCHAR(30),
    phone             VARCHAR(20),
    email             VARCHAR(100),
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_customers_city_state ON customers(city, state);
CREATE INDEX idx_customers_signup_date ON customers(signup_date);
CREATE INDEX idx_customers_segment ON customers(customer_segment);

-- ======================================================================
-- CATEGORIES
-- ======================================================================

CREATE TABLE categories (
    category_id       SERIAL PRIMARY KEY,
    category_name   VARCHAR(100) NOT NULL,
    subcategory     VARCHAR(100) NOT NULL,
    category_level   VARCHAR(20),
    description     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category_name, subcategory)
);

CREATE INDEX idx_categories_name ON categories(category_name);

-- ======================================================================
-- SUPPLIERS
-- ======================================================================

CREATE TABLE suppliers (
    supplier_id     SERIAL PRIMARY KEY,
    supplier_name VARCHAR(150) NOT NULL,
    contact_name  VARCHAR(100),
    city          VARCHAR(100),
    state         VARCHAR(100),
    country       VARCHAR(50) DEFAULT 'India',
    phone         VARCHAR(20),
    email         VARCHAR(100),
    rating        NUMERIC(3,2) DEFAULT 4.00 CHECK (rating >= 0 AND rating <= 5),
    lead_time_days INTEGER DEFAULT 7,
    reliability_score NUMERIC(5,2) DEFAULT 85.00,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_suppliers_name ON suppliers(supplier_name);

-- ======================================================================
-- PRODUCTS
-- ======================================================================

CREATE TABLE products (
    product_id      SERIAL PRIMARY KEY,
    product_name  VARCHAR(255) NOT NULL,
    category_id   INTEGER NOT NULL REFERENCES categories(category_id),
    supplier_id   INTEGER NOT NULL REFERENCES suppliers(supplier_id),
    sku_code      VARCHAR(50) UNIQUE,
    cost_price    NUMERIC(12,2) NOT NULL CHECK (cost_price >= 0),
    selling_price NUMERIC(12,2) NOT NULL CHECK (selling_price >= 0),
    launch_date   DATE NOT NULL,
    weight_kg     NUMERIC(8,2),
    product_status VARCHAR(20) DEFAULT 'Active' CHECK (product_status IN ('Active', 'Discontinued', 'Out of Season')),
    brand_name    VARCHAR(100),
    description   TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_supplier ON products(supplier_id);
CREATE INDEX idx_products_status ON products(product_status);
CREATE INDEX idx_products_price ON products(selling_price);

-- ======================================================================
-- STORES
-- ======================================================================

CREATE TABLE stores (
    store_id      SERIAL PRIMARY KEY,
    store_name    VARCHAR(100) NOT NULL UNIQUE,
    store_type    VARCHAR(30) DEFAULT 'Warehouse' CHECK (store_type IN ('Warehouse', 'Retail', 'Fulfillment Center')),
    city          VARCHAR(100) NOT NULL,
    state         VARCHAR(100) NOT NULL,
    country       VARCHAR(50) NOT NULL DEFAULT 'India',
    opening_date  DATE,
    store_area_sqft INTEGER,
    store_manager_id INTEGER,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_stores_city_state ON stores(city, state);

-- ======================================================================
-- EMPLOYEES
-- ======================================================================

CREATE TABLE employees (
    employee_id   SERIAL PRIMARY KEY,
    first_name    VARCHAR(50) NOT NULL,
    last_name     VARCHAR(50) NOT NULL,
    gender        CHAR(1) CHECK (gender IN ('M', 'F')),
    role          VARCHAR(50) NOT NULL,
    department    VARCHAR(50),
    store_id      INTEGER REFERENCES stores(store_id),
    hire_date     DATE NOT NULL,
    salary        NUMERIC(12,2),
    email         VARCHAR(100),
    phone         VARCHAR(20),
    performance_score NUMERIC(5,2) DEFAULT 75.00,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE stores ADD CONSTRAINT fk_stores_manager
    FOREIGN KEY (store_manager_id) REFERENCES employees(employee_id);

CREATE INDEX idx_employees_store ON employees(store_id);
CREATE INDEX idx_employees_role ON employees(role);

-- ======================================================================
-- INVENTORY
-- ======================================================================

CREATE TABLE inventory (
    inventory_id  BIGSERIAL PRIMARY KEY,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    store_id      INTEGER NOT NULL REFERENCES stores(store_id),
    stock_quantity INTEGER NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
    reorder_level INTEGER DEFAULT 50,
    safety_stock  INTEGER DEFAULT 30,
    last_restock_date DATE,
    average_daily_demand NUMERIC(10,2),
    lead_time_days INTEGER DEFAULT 7,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, store_id)
);

CREATE INDEX idx_inventory_product ON inventory(product_id);
CREATE INDEX idx_inventory_store ON inventory(store_id);
CREATE INDEX idx_inventory_stock ON inventory(stock_quantity);

-- ======================================================================
-- MARKETING CAMPAIGNS
-- ======================================================================

CREATE TABLE marketing_campaigns (
    campaign_id     SERIAL PRIMARY KEY,
    campaign_name VARCHAR(150) NOT NULL,
    campaign_type VARCHAR(50) NOT NULL,
    channel       VARCHAR(50) NOT NULL,
    start_date    DATE NOT NULL,
    end_date      DATE NOT NULL,
    target_audience VARCHAR(100),
    total_budget  NUMERIC(14,2) NOT NULL CHECK (total_budget >= 0),
    target_revenue NUMERIC(14,2),
    status        VARCHAR(20) DEFAULT 'Active' CHECK (status IN ('Active', 'Completed', 'Planned', 'Cancelled')),
    description   TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_marketing_campaigns_dates ON marketing_campaigns(start_date, end_date);
CREATE INDEX idx_marketing_campaigns_channel ON marketing_campaigns(channel);

-- ======================================================================
-- PAYMENTS
-- ======================================================================

CREATE TABLE payments (
    payment_id     BIGSERIAL PRIMARY KEY,
    customer_id   BIGINT NOT NULL REFERENCES customers(customer_id),
    payment_method VARCHAR(30) NOT NULL,
    payment_status VARCHAR(20) DEFAULT 'Success' CHECK (payment_status IN ('Success', 'Failed', 'Pending', 'Refunded')),
    transaction_date TIMESTAMP NOT NULL,
    amount        NUMERIC(14,2) NOT NULL CHECK (amount >= 0),
    card_last4   VARCHAR(4),
    bank_name     VARCHAR(100),
    upi_id          VARCHAR(100),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_payments_customer ON payments(customer_id);
CREATE INDEX idx_payments_date ON payments(transaction_date);
CREATE INDEX idx_payments_status ON payments(payment_status);
CREATE INDEX idx_payments_method ON payments(payment_method);

-- ======================================================================
-- ORDERS
-- ======================================================================

CREATE TABLE orders (
    order_id       BIGSERIAL PRIMARY KEY,
    customer_id   BIGINT NOT NULL REFERENCES customers(customer_id),
    order_date    TIMESTAMP NOT NULL,
    order_status  VARCHAR(20) NOT NULL CHECK (order_status IN ('Delivered', 'Cancelled', 'Returned', 'Processing', 'Shipped', 'Pending')),
    store_id      INTEGER REFERENCES stores(store_id),
    payment_id    BIGINT REFERENCES payments(payment_id),
    campaign_id   INTEGER REFERENCES marketing_campaigns(campaign_id),
    shipping_date DATE,
    delivery_date DATE,
    shipping_cost   NUMERIC(10,2) DEFAULT 0,
    discount_amount NUMERIC(12,2) DEFAULT 0,
    tax_amount    NUMERIC(12,2) DEFAULT 0,
    order_total   NUMERIC(14,2) NOT NULL CHECK (order_total >= 0),
    device_type   VARCHAR(20),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(order_status);
CREATE INDEX idx_orders_store ON orders(store_id);
CREATE INDEX idx_orders_payment ON orders(payment_id);
CREATE INDEX idx_orders_campaign ON orders(campaign_id);

-- ======================================================================
-- ORDER ITEMS
-- ======================================================================

CREATE TABLE order_items (
    order_item_id BIGSERIAL PRIMARY KEY,
    order_id      BIGINT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    unit_price    NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    discount      NUMERIC(12,2) DEFAULT 0 CHECK (discount >= 0),
    discount_pct      NUMERIC(5,2) DEFAULT 0,
    tax           NUMERIC(12,2) DEFAULT 0,
    line_total    NUMERIC(14,2) NOT NULL CHECK (line_total >= 0),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- ======================================================================
-- RETURNS
-- ======================================================================

CREATE TABLE returns (
    return_id     BIGSERIAL PRIMARY KEY,
    order_item_id BIGINT NOT NULL REFERENCES order_items(order_item_id),
    order_id      BIGINT NOT NULL REFERENCES orders(order_id),
    customer_id   BIGINT NOT NULL REFERENCES customers(customer_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    return_date   DATE NOT NULL,
    return_reason VARCHAR(100),
    return_status  VARCHAR(20) DEFAULT 'Processed' CHECK (return_status IN ('Requested', 'Approved', 'Processed', 'Rejected')),
    refund_amount NUMERIC(12,2) NOT NULL CHECK (refund_amount >= 0),
    quantity_returned INTEGER NOT NULL DEFAULT 1,
    processing_days INTEGER,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_returns_order ON returns(order_id);
CREATE INDEX idx_returns_customer ON returns(customer_id);
CREATE INDEX idx_returns_product ON returns(product_id);
CREATE INDEX idx_returns_date ON returns(return_date);

-- ======================================================================
-- REVIEWS
-- ======================================================================

CREATE TABLE reviews (
    review_id     BIGSERIAL PRIMARY KEY,
    customer_id   BIGINT NOT NULL REFERENCES customers(customer_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    order_id      BIGINT REFERENCES orders(order_id),
    review_date   DATE NOT NULL,
    rating        INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review_title  VARCHAR(200),
    review_text   TEXT,
    helpful_votes INTEGER DEFAULT 0,
    verified_purchase BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reviews_customer ON reviews(customer_id);
CREATE INDEX idx_reviews_product ON reviews(product_id);
CREATE INDEX idx_reviews_date ON reviews(review_date);
CREATE INDEX idx_reviews_rating ON reviews(rating);

-- ======================================================================
-- MARKETING SPEND (daily)
-- ======================================================================

CREATE TABLE marketing_spend (
    spend_id      BIGSERIAL PRIMARY KEY,
    campaign_id   INTEGER NOT NULL REFERENCES marketing_campaigns(campaign_id),
    spend_date    DATE NOT NULL,
    channel       VARCHAR(50) NOT NULL,
    impressions   BIGINT DEFAULT 0,
    clicks        INTEGER DEFAULT 0,
    spend_amount  NUMERIC(12,2) NOT NULL CHECK (spend_amount >= 0),
    ctr           NUMERIC(8,6),
    cpc           NUMERIC(10,4),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(campaign_id, spend_date, channel)
);

CREATE INDEX idx_marketing_spend_campaign ON marketing_spend(campaign_id);
CREATE INDEX idx_marketing_spend_date ON marketing_spend(spend_date);
CREATE INDEX idx_marketing_spend_channel ON marketing_spend(channel);

-- ======================================================================
-- WEBSITE SESSIONS
-- ======================================================================

CREATE TABLE website_sessions (
    session_id    BIGSERIAL PRIMARY KEY,
    customer_id   BIGINT REFERENCES customers(customer_id),
    session_date  DATE NOT NULL,
    session_start TIMESTAMP NOT NULL,
    session_end   TIMESTAMP,
    device_type   VARCHAR(20),
    channel       VARCHAR(50),
    campaign_id   INTEGER REFERENCES marketing_campaigns(campaign_id),
    page_views    INTEGER DEFAULT 0,
    product_views     INTEGER DEFAULT 0,
    cart_adds      INTEGER DEFAULT 0,
    checkout_started INTEGER DEFAULT 0,
    checkout_completed INTEGER DEFAULT 0,
    session_duration_sec INTEGER DEFAULT 0,
    bounce_rate   BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_website_sessions_customer ON website_sessions(customer_id);
CREATE INDEX idx_website_sessions_date ON website_sessions(session_date);
CREATE INDEX idx_website_sessions_channel ON website_sessions(channel);
CREATE INDEX idx_website_sessions_campaign ON website_sessions(campaign_id);
