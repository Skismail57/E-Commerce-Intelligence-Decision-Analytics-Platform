# Data Model Documentation

## Overview

The E-Commerce Intelligence Platform uses a star schema with dimensional modeling optimized for analytics and reporting. The data warehouse is built on PostgreSQL with 15 core tables supporting comprehensive e-commerce analytics.

## Entity Relationship Diagram

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  customers  │──────<│   orders    │──────<│ order_items │
└─────────────┘       └─────────────┘       └─────────────┘
       │                     │                       │
       │                     │                       │
       │              ┌──────┴──────┐               │
       │              │             │               │
       │         ┌────▼────┐  ┌────▼────┐     ┌────▼────┐
       │         │ stores  │  │payments │     │products │
       │         └─────────┘  └─────────┘     └─────────┘
       │                                           │
       │                                    ┌──────┴──────┐
       │                                    │             │
       │                               ┌────▼────┐  ┌────▼────┐
       │                               │categories│ │suppliers│
       │                               └─────────┘  └─────────┘
       │
┌──────▼──────┐
│ marketing_  │
│ campaigns   │
└─────────────┘
       │
┌──────▼──────┐
│ marketing_   │
│ spend       │
└─────────────┘

┌─────────────┐       ┌─────────────┐
│website_     │       │  returns    │
│sessions     │──────>│             │
└─────────────┘       └─────────────┘
       │
┌──────▼──────┐
│  reviews    │
└─────────────┘

┌─────────────┐       ┌─────────────┐
│  inventory  │──────>│  employees  │
└─────────────┘       └─────────────┘
```

## Core Tables

### 1. customers

Customer master table with demographic and segmentation data.

| Column | Type | Description |
|--------|------|-------------|
| customer_id | BIGSERIAL | Primary key |
| first_name | VARCHAR(50) | Customer first name |
| last_name | VARCHAR(50) | Customer last name |
| gender | CHAR(1) | M/F |
| date_of_birth | DATE | Customer DOB |
| age | INTEGER | Calculated age |
| city | VARCHAR(100) | Customer city |
| state | VARCHAR(100) | Customer state |
| country | VARCHAR(50) | Customer country (default: India) |
| signup_date | DATE | Registration date |
| customer_segment | VARCHAR(30) | Business segment (Regular, Premium, VIP) |
| signup_channel | VARCHAR(30) | Acquisition channel |
| phone | VARCHAR(20) | Contact phone |
| email | VARCHAR(100) | Contact email |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

**Indexes:**
- idx_customers_city_state (city, state)
- idx_customers_signup_date (signup_date)
- idx_customers_segment (customer_segment)

### 2. products

Product master with pricing and supplier information.

| Column | Type | Description |
|--------|------|-------------|
| product_id | SERIAL | Primary key |
| product_name | VARCHAR(255) | Product name |
| category_id | INTEGER | FK to categories |
| supplier_id | INTEGER | FK to suppliers |
| sku_code | VARCHAR(50) | Unique SKU |
| cost_price | NUMERIC(12,2) | Cost price |
| selling_price | NUMERIC(12,2) | Selling price |
| launch_date | DATE | Product launch date |
| weight_kg | NUMERIC(8,2) | Product weight |
| product_status | VARCHAR(20) | Active/Discontinued/Out of Season |
| brand_name | VARCHAR(100) | Brand name |
| description | TEXT | Product description |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

**Indexes:**
- idx_products_category (category_id)
- idx_products_supplier (supplier_id)
- idx_products_status (product_status)
- idx_products_price (selling_price)

### 3. orders

Order header table with order-level information.

| Column | Type | Description |
|--------|------|-------------|
| order_id | BIGSERIAL | Primary key |
| customer_id | BIGINT | FK to customers |
| order_date | TIMESTAMP | Order timestamp |
| order_status | VARCHAR(20) | Delivered/Cancelled/Returned/Processing/Shipped/Pending |
| store_id | INTEGER | FK to stores |
| payment_id | BIGINT | FK to payments |
| campaign_id | INTEGER | FK to marketing_campaigns |
| shipping_date | DATE | Shipping date |
| delivery_date | DATE | Delivery date |
| shipping_cost | NUMERIC(10,2) | Shipping charges |
| discount_amount | NUMERIC(12,2) | Order-level discount |
| tax_amount | NUMERIC(12,2) | Tax amount |
| order_total | NUMERIC(14,2) | Total order value |
| device_type | VARCHAR(20) | Mobile/Desktop/Tablet |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

**Indexes:**
- idx_orders_customer (customer_id)
- idx_orders_date (order_date)
- idx_orders_status (order_status)
- idx_orders_store (store_id)
- idx_orders_payment (payment_id)
- idx_orders_campaign (campaign_id)

### 4. order_items

Order line items with product-level details.

| Column | Type | Description |
|--------|------|-------------|
| order_item_id | BIGSERIAL | Primary key |
| order_id | BIGINT | FK to orders |
| product_id | INTEGER | FK to products |
| quantity | INTEGER | Quantity ordered |
| unit_price | NUMERIC(12,2) | Unit price |
| discount | NUMERIC(12,2) | Line discount |
| discount_pct | NUMERIC(5,2) | Discount percentage |
| tax | NUMERIC(12,2) | Line tax |
| line_total | NUMERIC(14,2) | Line total |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_order_items_order (order_id)
- idx_order_items_product (product_id)

### 5. categories

Product category hierarchy.

| Column | Type | Description |
|--------|------|-------------|
| category_id | SERIAL | Primary key |
| category_name | VARCHAR(100) | Category name |
| subcategory | VARCHAR(100) | Subcategory name |
| category_level | VARCHAR(20) | Category level |
| description | TEXT | Category description |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_categories_name (category_name)

### 6. suppliers

Supplier master table.

| Column | Type | Description |
|--------|------|-------------|
| supplier_id | SERIAL | Primary key |
| supplier_name | VARCHAR(150) | Supplier name |
| contact_name | VARCHAR(100) | Contact person |
| city | VARCHAR(100) | Supplier city |
| state | VARCHAR(100) | Supplier state |
| country | VARCHAR(50) | Supplier country |
| phone | VARCHAR(20) | Contact phone |
| email | VARCHAR(100) | Contact email |
| rating | NUMERIC(3,2) | Supplier rating (0-5) |
| lead_time_days | INTEGER | Standard lead time |
| reliability_score | NUMERIC(5,2) | Reliability score |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_suppliers_name (supplier_name)

### 7. stores

Store/warehouse locations.

| Column | Type | Description |
|--------|------|-------------|
| store_id | SERIAL | Primary key |
| store_name | VARCHAR(100) | Store name |
| store_type | VARCHAR(30) | Warehouse/Retail/Fulfillment Center |
| city | VARCHAR(100) | Store city |
| state | VARCHAR(100) | Store state |
| country | VARCHAR(50) | Store country |
| opening_date | DATE | Store opening date |
| store_area_sqft | INTEGER | Store area |
| store_manager_id | INTEGER | FK to employees |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_stores_city_state (city, state)

### 8. inventory

Inventory levels by product and store.

| Column | Type | Description |
|--------|------|-------------|
| inventory_id | BIGSERIAL | Primary key |
| product_id | INTEGER | FK to products |
| store_id | INTEGER | FK to stores |
| stock_quantity | INTEGER | Current stock |
| reorder_level | INTEGER | Reorder trigger level |
| safety_stock | INTEGER | Safety stock level |
| last_restock_date | DATE | Last restock date |
| average_daily_demand | NUMERIC(10,2) | Avg daily demand |
| lead_time_days | INTEGER | Lead time in days |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

**Indexes:**
- idx_inventory_product (product_id)
- idx_inventory_store (store_id)
- idx_inventory_stock (stock_quantity)

### 9. payments

Payment transaction details.

| Column | Type | Description |
|--------|------|-------------|
| payment_id | BIGSERIAL | Primary key |
| customer_id | BIGINT | FK to customers |
| payment_method | VARCHAR(30) | Payment method |
| payment_status | VARCHAR(20) | Success/Failed/Pending/Refunded |
| transaction_date | TIMESTAMP | Transaction timestamp |
| amount | NUMERIC(14,2) | Payment amount |
| card_last4 | VARCHAR(4) | Card last 4 digits |
| bank_name | VARCHAR(100) | Bank name |
| upi_id | VARCHAR(100) | UPI ID |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_payments_customer (customer_id)
- idx_payments_date (transaction_date)
- idx_payments_status (payment_status)
- idx_payments_method (payment_method)

### 10. returns

Product return information.

| Column | Type | Description |
|--------|------|-------------|
| return_id | BIGSERIAL | Primary key |
| order_item_id | BIGINT | FK to order_items |
| order_id | BIGINT | FK to orders |
| customer_id | BIGINT | FK to customers |
| product_id | INTEGER | FK to products |
| return_date | DATE | Return date |
| return_reason | VARCHAR(100) | Return reason |
| return_status | VARCHAR(20) | Requested/Approved/Processed/Rejected |
| refund_amount | NUMERIC(12,2) | Refund amount |
| quantity_returned | INTEGER | Quantity returned |
| processing_days | INTEGER | Processing time |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_returns_order (order_id)
- idx_returns_customer (customer_id)
- idx_returns_product (product_id)
- idx_returns_date (return_date)

### 11. reviews

Customer product reviews.

| Column | Type | Description |
|--------|------|-------------|
| review_id | BIGSERIAL | Primary key |
| customer_id | BIGINT | FK to customers |
| product_id | INTEGER | FK to products |
| order_id | BIGINT | FK to orders |
| review_date | DATE | Review date |
| rating | INTEGER | Rating (1-5) |
| review_title | VARCHAR(200) | Review title |
| review_text | TEXT | Review content |
| helpful_votes | INTEGER | Helpful votes |
| verified_purchase | BOOLEAN | Verified purchase flag |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_reviews_customer (customer_id)
- idx_reviews_product (product_id)
- idx_reviews_date (review_date)
- idx_reviews_rating (rating)

### 12. marketing_campaigns

Marketing campaign details.

| Column | Type | Description |
|--------|------|-------------|
| campaign_id | SERIAL | Primary key |
| campaign_name | VARCHAR(150) | Campaign name |
| campaign_type | VARCHAR(50) | Campaign type |
| channel | VARCHAR(50) | Marketing channel |
| start_date | DATE | Campaign start |
| end_date | DATE | Campaign end |
| target_audience | VARCHAR(100) | Target audience |
| total_budget | NUMERIC(14,2) | Campaign budget |
| target_revenue | NUMERIC(14,2) | Target revenue |
| status | VARCHAR(20) | Active/Completed/Planned/Cancelled |
| description | TEXT | Campaign description |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_marketing_campaigns_dates (start_date, end_date)
- idx_marketing_campaigns_channel (channel)

### 13. marketing_spend

Daily marketing spend by channel.

| Column | Type | Description |
|--------|------|-------------|
| spend_id | BIGSERIAL | Primary key |
| campaign_id | INTEGER | FK to marketing_campaigns |
| spend_date | DATE | Spend date |
| channel | VARCHAR(50) | Marketing channel |
| impressions | BIGINT | Ad impressions |
| clicks | INTEGER | Ad clicks |
| spend_amount | NUMERIC(12,2) | Spend amount |
| ctr | NUMERIC(8,6) | Click-through rate |
| cpc | NUMERIC(10,4) | Cost per click |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_marketing_spend_campaign (campaign_id)
- idx_marketing_spend_date (spend_date)
- idx_marketing_spend_channel (channel)

### 14. website_sessions

Website session tracking.

| Column | Type | Description |
|--------|------|-------------|
| session_id | BIGSERIAL | Primary key |
| customer_id | BIGINT | FK to customers |
| session_date | DATE | Session date |
| session_start | TIMESTAMP | Session start |
| session_end | TIMESTAMP | Session end |
| device_type | VARCHAR(20) | Device type |
| channel | VARCHAR(50) | Acquisition channel |
| campaign_id | INTEGER | FK to marketing_campaigns |
| page_views | INTEGER | Page views |
| product_views | INTEGER | Product views |
| cart_adds | INTEGER | Cart additions |
| checkout_started | INTEGER | Checkout started |
| checkout_completed | INTEGER | Checkout completed |
| session_duration_sec | INTEGER | Session duration |
| bounce_rate | BOOLEAN | Bounce flag |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_website_sessions_customer (customer_id)
- idx_website_sessions_date (session_date)
- idx_website_sessions_channel (channel)
- idx_website_sessions_campaign (campaign_id)

### 15. employees

Employee information for operations analytics.

| Column | Type | Description |
|--------|------|-------------|
| employee_id | SERIAL | Primary key |
| first_name | VARCHAR(50) | First name |
| last_name | VARCHAR(50) | Last name |
| gender | CHAR(1) | M/F |
| role | VARCHAR(50) | Job role |
| department | VARCHAR(50) | Department |
| store_id | INTEGER | FK to stores |
| hire_date | DATE | Hire date |
| salary | NUMERIC(12,2) | Salary |
| email | VARCHAR(100) | Email |
| phone | VARCHAR(20) | Phone |
| performance_score | NUMERIC(5,2) | Performance score |
| created_at | TIMESTAMP | Record creation timestamp |

**Indexes:**
- idx_employees_store (store_id)
- idx_employees_role (role)

## Key Views

### Denormalized Views

1. **vw_customer_360**: Complete customer profile with lifetime metrics
2. **vw_product_360**: Complete product profile with performance metrics
3. **vw_order_line_fact**: Denormalized transaction fact table
4. **vw_daily_sales**: Daily aggregated sales metrics
5. **vw_marketing_channel_funnel**: Marketing funnel by channel
6. **vw_inventory_health**: Inventory status and recommendations
7. **vw_rfm_segments**: RFM analysis with segment classification

### Analytics Views

1. **vw_analytics_cohort_retention**: Cohort retention matrix
2. **vw_analytics_product_matrix**: Product quadrant analysis
3. **vw_ml_churn_features**: ML-ready churn feature set
4. **vw_ml_demand_forecast_train**: Forecasting training data

### Transformation Views

1. **vw_inventory_analytics**: Advanced inventory metrics
2. **vw_demand_daily**: Daily demand with calendar features
3. **vw_anomaly_prep**: Anomaly detection preparation
4. **vw_supplier_performance**: Supplier 360° metrics
5. **vw_store_performance**: Store performance metrics

### KPI Views

1. **kpi_executive_snapshot**: Single-row executive KPI summary
2. **kpi_monthly_trend**: Monthly KPI trends
3. **kpi_region_performance**: Regional performance breakdown
4. **kpi_category_performance**: Category performance breakdown
5. **kpi_customer_lifetime**: CLV cohort analysis
6. **kpi_top_10_customers**: Top customers by profit
7. **kpi_payment_methods**: Payment method analysis

## Data Relationships

### Primary Relationships

- **customers → orders**: One-to-many (customer can have many orders)
- **orders → order_items**: One-to-many (order can have many items)
- **products → order_items**: One-to-many (product can be in many orders)
- **categories → products**: One-to-many (category has many products)
- **suppliers → products**: One-to-many (supplier supplies many products)
- **stores → orders**: One-to-many (store fulfills many orders)
- **stores → inventory**: One-to-many (store has many inventory records)
- **products → inventory**: One-to-many (product in many stores)
- **marketing_campaigns → orders**: One-to-many (campaign drives many orders)
- **payments → orders**: One-to-one (payment per order)
- **customers → website_sessions**: One-to-many (customer has many sessions)
- **customers → reviews**: One-to-many (customer writes many reviews)

### Cascade Rules

- **order_items**: CASCADE DELETE on orders
- **returns**: No cascade (audit trail)
- **reviews**: No cascade (audit trail)

## Data Quality Rules

### Validation Rules

1. **Numeric Constraints**
   - All monetary fields ≥ 0
   - All quantity fields ≥ 1
   - Ratings between 1-5
   - Dates within logical ranges

2. **Referential Integrity**
   - All foreign keys must reference valid parent records
   - Orders must have valid customer_id
   - Order items must have valid product_id

3. **Business Rules**
   - Selling price ≥ cost price
   - Order total = sum(line totals) + shipping - discounts
   - Stock quantity ≥ 0
   - Lead time days ≥ 1

### Data Quality Checks

- **Completeness**: All required fields populated
- **Accuracy**: Values within expected ranges
- **Consistency**: Related fields align logically
- **Timeliness**: Data freshness within SLA
- **Uniqueness**: No duplicate primary keys

## Performance Considerations

### Indexing Strategy

- **Foreign keys**: All FK columns indexed
- **Date columns**: All date columns indexed for time-series queries
- **Status columns**: Indexed for filtering
- **Composite indexes**: For common query patterns

### Partitioning

Consider partitioning for large tables:
- **orders**: Range partition by order_date (monthly)
- **order_items**: Reference partition on orders
- **website_sessions**: Range partition by session_date (monthly)

### Query Optimization

- Use materialized views for heavy aggregations
- Implement query caching for frequently accessed KPIs
- Use connection pooling for high concurrency

## Data Lineage

```
Raw Data (CSV) → Staging (Cleaned) → Warehouse (Structured) → Analytics (Transformed) → Reports (Visualized)
```

### Transformation Stages

1. **Ingestion**: Raw data loaded to staging
2. **Cleaning**: Data quality checks and corrections
3. **Validation**: Schema validation and business rules
4. **Transformation**: Feature engineering and aggregations
5. **Loading**: Final data loaded to warehouse
6. **Analytics**: KPI calculations and ML features

## Security Considerations

### Access Control

- **Read-only**: Analytics users
- **Read-write**: ETL processes
- **Admin**: Data engineering team

### Data Masking

- **PII**: Customer emails, phone numbers masked for non-admin users
- **Financial**: Cost prices restricted to authorized users
- **Sensitive**: Employee salaries restricted to HR

### Audit Trail

- All DML operations logged
- User access tracked
- Data changes audited

---

**Document Version**: 1.0  
**Last Updated**: September 2026  
**Maintained By**: Data Engineering Team
