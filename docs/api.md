# API Documentation

## FastAPI REST API

The E-Commerce Intelligence Platform provides a REST API built with FastAPI for programmatic access to analytics and insights.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication. In production, implement JWT-based authentication.

## Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-09-02T12:00:00Z",
  "version": "1.0.0"
}
```

### Customer Analytics

#### Get Customer 360

**GET** `/customers/{customer_id}`

Retrieve complete customer profile with lifetime metrics.

**Parameters:**
- `customer_id` (path): Customer ID

**Response:**
```json
{
  "customer_id": 12345,
  "first_name": "John",
  "last_name": "Doe",
  "total_orders": 15,
  "total_revenue": 45000.00,
  "avg_order_value": 3000.00,
  "rfm_segment": "Loyal Customers",
  "clv": 125000.00,
  "churn_risk": "Low",
  "days_since_last_order": 12
}
```

#### Get RFM Segments

**GET** `/customers/rfm/segments`

Retrieve RFM segment distribution.

**Query Parameters:**
- `segment` (optional): Filter by segment name

**Response:**
```json
{
  "segments": [
    {
      "segment": "Champions",
      "count": 12500,
      "percentage": 13.0,
      "avg_spend": 85000.00
    },
    {
      "segment": "Loyal Customers",
      "count": 25000,
      "percentage": 26.0,
      "avg_spend": 45000.00
    }
  ]
}
```

#### Get Churn Risk

**GET** `/customers/churn/risk`

Retrieve customers at churn risk.

**Query Parameters:**
- `risk_threshold` (optional): Minimum churn probability (default: 0.5)
- `limit` (optional): Maximum results (default: 100)

**Response:**
```json
{
  "at_risk_customers": [
    {
      "customer_id": 12345,
      "churn_probability": 0.87,
      "risk_tier": "High",
      "total_spend": 75000.00,
      "days_since_last_order": 143,
      "reasons": [
        "No purchase for 143 days",
        "Order frequency declining",
        "High discount dependency"
      ]
    }
  ],
  "total_at_risk": 4821,
  "high_risk_count": 1131
}
```

### Product Analytics

#### Get Product矩阵

**GET** `/products/matrix`

Retrieve product quadrant analysis.

**Query Parameters:**
- `quadrant` (optional): Filter by quadrant (Stars, Volume, Remove, Premium)
- `category` (optional): Filter by category

**Response:**
```json
{
  "products": [
    {
      "product_id": 123,
      "product_name": "Wireless Headphones",
      "category": "Electronics",
      "revenue_inr": 5000000.00,
      "profit_inr": 1500000.00,
      "margin_pct": 30.0,
      "units_sold": 2500,
      "quadrant": "Stars"
    }
  ],
  "quadrant_summary": {
    "Stars": 125,
    "Volume": 340,
    "Remove": 89,
    "Premium": 446
  }
}
```

#### Get Inventory Status

**GET** `/products/inventory/status`

Retrieve inventory health status.

**Query Parameters:**
- `status` (optional): Filter by status (Out of Stock, Critical, Reorder, Healthy, Overstock)

**Response:**
```json
{
  "inventory": [
    {
      "product_id": 123,
      "product_name": "Wireless Headphones",
      "current_stock": 15,
      "daily_demand": 31,
      "days_of_stock": 0.5,
      "reorder_quantity": 500,
      "urgency": "Critical",
      "status": "Out of Stock Risk"
    }
  ],
  "summary": {
    "total_skus": 5000,
    "out_of_stock": 12,
    "critical": 45,
    "reorder": 234,
    "healthy": 4500,
    "overstock": 209
  }
}
```

### Sales Analytics

#### Get Daily Sales

**GET** `/sales/daily`

Retrieve daily sales metrics.

**Query Parameters:**
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Response:**
```json
{
  "daily_sales": [
    {
      "date": "2024-09-01",
      "revenue_inr": 3500000.00,
      "orders": 520,
      "units_sold": 1250,
      "avg_order_value": 6730.77
    }
  ],
  "summary": {
    "total_revenue": 84200000.00,
    "total_orders": 184521,
    "avg_daily_revenue": 2800000.00
  }
}
```

#### Get Revenue Trend

**GET** `/sales/revenue/trend`

Retrieve revenue trend analysis.

**Query Parameters:**
- `period` (optional): Time period (daily, weekly, monthly)
- `months` (optional): Number of months (default: 12)

**Response:**
```json
{
  "trend": [
    {
      "period": "2024-01",
      "revenue_inr": 7500000.00,
      "orders": 15000,
      "growth_pct": 12.5
    }
  ],
  "yoy_growth": 15.2,
  "mom_growth": 2.3
}
```

### Marketing Analytics

#### Get Campaign Performance

**GET** `/marketing/campaigns/performance`

Retrieve marketing campaign performance.

**Query Parameters:**
- `channel` (optional): Filter by channel
- `status` (optional): Filter by status

**Response:**
```json
{
  "campaigns": [
    {
      "campaign_id": 1,
      "campaign_name": "Diwali Sale 2024",
      "channel": "Google",
      "total_spend": 500000.00,
      "impressions": 1000000,
      "clicks": 25000,
      "ctr_pct": 2.5,
      "orders": 500,
      "revenue_inr": 2500000.00,
      "cac": 1000.00,
      "roas": 5.0,
      "status": "Active"
    }
  ]
}
```

#### Get Marketing Funnel

**GET** `/marketing/funnel`

Retrieve marketing funnel metrics.

**Query Parameters:**
- `channel` (optional): Filter by channel
- `start_date` (optional): Start date
- `end_date` (optional): End date

**Response:**
```json
{
  "funnel": {
    "impressions": 1000000,
    "clicks": 25000,
    "sessions": 15000,
    "product_views": 8000,
    "cart_adds": 3000,
    "checkouts_started": 2000,
    "checkouts_completed": 1500,
    "orders": 1200
  },
  "conversion_rates": {
    "ctr": 2.5,
    "click_to_session": 60.0,
    "session_to_cart": 20.0,
    "cart_to_checkout": 66.7,
    "checkout_to_order": 80.0,
    "overall": 0.12
  }
}
```

### Forecasting

#### Get Demand Forecast

**GET** `/forecasting/demand`

Retrieve demand forecast for products.

**Query Parameters:**
- `product_id` (optional): Specific product ID
- `category` (optional): Filter by category
- `horizon_days` (optional): Forecast horizon (default: 30)
- `model` (optional): Forecasting model (prophet, arima, xgboost)

**Response:**
```json
{
  "forecast": [
    {
      "date": "2024-09-03",
      "product_id": 123,
      "forecasted_demand": 35,
      "lower_bound": 28,
      "upper_bound": 42,
      "confidence": 0.95
    }
  ],
  "model_metrics": {
    "mae": 4.2,
    "rmse": 6.8,
    "mape": 8.5,
    "model_type": "prophet"
  }
}
```

### Anomaly Detection

#### Get Anomalies

**GET** `/anomalies`

Retrieve detected anomalies.

**Query Parameters:**
- `type` (optional): Anomaly type (revenue, orders, inventory)
- `severity` (optional): Severity level (low, medium, high)
- `start_date` (optional): Start date
- `end_date` (optional): End date

**Response:**
```json
{
  "anomalies": [
    {
      "date": "2024-09-01",
      "type": "revenue",
      "severity": "high",
      "value": 1500000.00,
      "expected": 3500000.00,
      "deviation_pct": -57.1,
      "z_score": -3.8,
      "description": "Revenue drop detected in Electronics category"
    }
  ],
  "summary": {
    "total_anomalies": 15,
    "high_severity": 3,
    "medium_severity": 7,
    "low_severity": 5
  }
}
```

### Decision Engine

#### Get Recommendations

**GET** `/decisions/recommendations`

Retrieve automated business recommendations.

**Query Parameters:**
- `category` (optional): Recommendation category (inventory, pricing, marketing, operations)
- `priority` (optional): Priority level (high, medium, low)

**Response:**
```json
{
  "recommendations": [
    {
      "id": "REC-001",
      "category": "inventory",
      "priority": "high",
      "title": "Replenish Electronics inventory",
      "description": "12 products at risk of stock-out within 7 days",
      "impact_score": 0.85,
      "estimated_impact": "Prevent ₹2.5 Cr potential revenue loss",
      "action_items": [
        "Initiate emergency replenishment for Wireless Headphones",
        "Review supplier lead times",
        "Consider safety stock adjustment"
      ],
      "created_at": "2024-09-02T12:00:00Z"
    }
  ]
}
```

#### Get Alerts

**GET** `/decisions/alerts`

Retrieve active business alerts.

**Query Parameters:**
- `type` (optional): Alert type (revenue, inventory, customer)
- `status` (optional): Alert status (active, acknowledged, resolved)

**Response:**
```json
{
  "alerts": [
    {
      "id": "ALT-001",
      "type": "revenue",
      "severity": "high",
      "title": "Revenue Drop Detected",
      "description": "Electronics category revenue down 17.2% this week",
      "potential_causes": [
        "Inventory shortage",
        "Increased returns",
        "Reduced traffic"
      ],
      "recommended_actions": [
        "Check inventory levels",
        "Review return rates",
        "Analyze traffic sources"
      ],
      "created_at": "2024-09-02T10:00:00Z",
      "status": "active"
    }
  ],
  "summary": {
    "total_alerts": 8,
    "high_severity": 2,
    "medium_severity": 4,
    "low_severity": 2
  }
}
```

### Executive KPIs

#### Get Executive Snapshot

**GET** `/kpis/executive`

Retrieve executive KPI snapshot.

**Response:**
```json
{
  "revenue": {
    "total_revenue_inr": 84200000.00,
    "net_revenue_inr": 79100000.00,
    "growth_pct": 12.5
  },
  "profit": {
    "gross_profit_inr": 23100000.00,
    "gross_margin_pct": 29.2,
    "growth_pct": 8.7
  },
  "orders": {
    "total_orders": 184521,
    "units_sold": 456789,
    "aov_inr": 4297.00
  },
  "customers": {
    "total_customers": 96340,
    "new_customers": 12345,
    "returning_customers": 83995
  },
  "efficiency": {
    "return_rate_pct": 6.8,
    "cac_inr": 1250.00,
    "roas": 4.2,
    "conversion_rate_pct": 3.5
  }
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details"
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default**: 100 requests per minute
- **Burst**: 200 requests per minute
- **Headers**: Rate limit info returned in response headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1633123456
```

## Pagination

List endpoints support pagination:

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 50, max: 1000)

**Response Headers:**
```
X-Total-Count: 1000
X-Total-Pages: 20
X-Current-Page: 1
```

## Response Format

All successful responses follow this structure:

```json
{
  "data": { /* Response data */ },
  "meta": {
    "timestamp": "2024-09-02T12:00:00Z",
    "request_id": "req_abc123",
    "version": "1.0.0"
  }
}
```

## SDK Examples

### Python

```python
import requests

base_url = "http://localhost:8000/api/v1"

# Get customer 360
response = requests.get(f"{base_url}/customers/12345")
customer_data = response.json()

# Get churn risk
response = requests.get(f"{base_url}/customers/churn/risk?risk_threshold=0.7")
churn_data = response.json()

# Get recommendations
response = requests.get(f"{base_url}/decisions/recommendations?priority=high")
recommendations = response.json()
```

### JavaScript

```javascript
const baseUrl = 'http://localhost:8000/api/v1';

// Get customer 360
fetch(`${baseUrl}/customers/12345`)
  .then(response => response.json())
  .then(data => console.log(data));

// Get churn risk
fetch(`${baseUrl}/customers/churn/risk?risk_threshold=0.7`)
  .then(response => response.json())
  .then(data => console.log(data));
```

## Webhooks

The platform supports webhooks for real-time notifications:

### Webhook Events

- `customer.churn_risk`: Customer churn risk change
- `inventory.stock_out`: Inventory stock-out event
- `revenue.anomaly`: Revenue anomaly detected
- `decision.recommendation`: New recommendation generated

### Webhook Configuration

Configure webhooks via environment variables or API:

```bash
WEBHOOK_URL=https://your-webhook-endpoint.com
WEBHOOK_SECRET=your-secret-key
WEBHOOK_EVENTS=customer.churn_risk,inventory.stock_out
```

## OpenAPI Specification

The complete OpenAPI specification is available at:

```
http://localhost:8000/docs
```

This interactive documentation allows you to test all endpoints directly.

---

**API Version**: 1.0.0  
**Last Updated**: September 2026  
**Base URL**: http://localhost:8000/api/v1
