# E-Commerce Intelligence & Decision Analytics Platform

![Cover Image](project%20Screenshots/Cover%20Image.png)

A production-style end-to-end data analytics platform that transforms raw e-commerce data into actionable business intelligence through advanced analytics, machine learning, and automated decision recommendations.

## 🚀 Overview

This platform demonstrates a complete data analytics workflow including:
- **Data Engineering**: ETL pipelines, data quality validation, PostgreSQL data warehouse
- **Advanced SQL Analytics**: Executive KPIs, cohort analysis, product matrix, churn features
- **Python Analytics**: Customer intelligence (RFM, CLV), product analytics, marketing analytics
- **Machine Learning**: Churn prediction, demand forecasting, anomaly detection
- **Decision Engine**: Automated business recommendations and alerting
- **Visualization**: Interactive Streamlit application with neon-themed UI

## 🏗️ Architecture

```
┌──────────────────────┐
│     RAW DATA         │
│ CSV / API / Database │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   DATA INGESTION     │
│      Python          │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ CLEANING & VALIDATION│
│ Pandas / Polars      │
│ Pandera              │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│    DATA WAREHOUSE    │
│     PostgreSQL       │
└──────────┬───────────┘
           ↓
┌────────────────┼────────────────┐
↓                ↓                ↓
Advanced SQL      Statistical        ML Models
Analytics         Analytics
↓                ↓                ↓
└────────────────┼────────────────┘
           ↓
┌─────────────────────────┐
│ ANALYTICS / DECISION    │
│        ENGINE           │
└────────────┬────────────┘
             ↓
┌────────────────┴────────────────┐
↓                                 ↓
Streamlit App (Neon UI)           FastAPI
↓                                 ↓
└────────────────┬────────────────┘
             ↓
BUSINESS RECOMMENDATIONS
```

## 📊 Application Screenshots & Features

### 🏠 Executive Dashboard

![Executive Dashboard KPI Metrics](project%20Screenshots/Executive_Dashboard_KPI_Metrics.png)

**Key Features:**
- **Real-time KPI Metrics**: Total Revenue, Net Revenue, Gross Profit, Gross Margin
- **Order Analytics**: Total orders, units sold, average order value
- **Customer Metrics**: Customer count, CAC, CLV, conversion rate
- **Performance Indicators**: Return rate, churn rate, ROAS
- **Advanced Filters**: Date range, order status, view mode selection
- **Interactive Charts**: Revenue trends, category performance, regional analysis

**Technology Used:**
- Plotly for interactive visualizations
- Pandas for data aggregation
- Streamlit for UI components

---

### 📈 Sales Analytics

![Sales Analytics Daily Performance](project%20Screenshots/Sales_Analytics_Daily_Performance.png)

![Sales Analytics Order Distribution Weekly](project%20Screenshots/Sales_Analytics_Order_Distribution_Weekly.png)

**Key Features:**
- **Daily Sales Trends**: Line chart showing revenue patterns over time
- **Order Status Distribution**: Pie chart of order statuses (Delivered, Shipped, Processing, Returned, Cancelled)
- **Day-of-Week Analysis**: Bar chart showing sales performance by day
- **Date Range Filtering**: Custom date selection for analysis
- **Revenue Breakdown**: Category-wise revenue analysis

**Technology Used:**
- Plotly Express for charts
- Pandas datetime operations
- Custom neon-themed styling

---

### 👥 Customer Analytics

![Customer Analytics RFM Distribution](project%20Screenshots/Customer_Analytics_RFM_Distribution.png)

![Customer Analytics RFM Table](project%20Screenshots/Customer_Analytics_RFM_Table.png)

![Customer Analytics CLV Value Tiers](project%20Screenshots/Customer_Analytics_CLV_Value_Tiers.png)

**Key Features:**
- **RFM Segmentation**: 
  - Champions (High recency, frequency, monetary)
  - Loyal Customers (High frequency, monetary)
  - At Risk (Low recency, high frequency)
  - Lost Customers (Low recency, frequency, monetary)
- **Customer Lifetime Value (CLV)**:
  - BG/NBD probabilistic modeling
  - Gamma-Gamma monetary value prediction
  - Value tier classification (High, Medium, Low)
- **Segment Distribution**: Pie chart showing customer segment breakdown
- **Detailed Segment Table**: Complete RFM metrics per segment
- **CLV Distribution**: Histogram of customer lifetime values

**Technology Used:**
- RFM analysis algorithms
- BG/NBD + Gamma-Gamma models (lifetimes library)
- Pandera for data validation
- Plotly for visualizations

---

### 🛍️ Product Analytics

![Product Analytics Performance Matrix](project%20Screenshots/Product_Analytics_Performance_Matrix.png)

![Product Analytics Quadrant Summary](project%20Screenshots/Product_Analytics_Quadrant_Summary.png)

![Product Analytics Top 10 Revenue](project%20Screenshots/Product_Analytics_Top_10_Revenue.png)

**Key Features:**
- **Product Matrix (BCG Analysis)**:
  - **Stars**: High growth, high market share
  - **Cash Cows**: Low growth, high market share
  - **Question Marks**: High growth, low market share
  - **Dogs**: Low growth, low market share
- **Quadrant Summary**: Product count, total revenue, total profit per quadrant
- **Top 10 Products by Revenue**: Ranked list with product names and revenue
- **Margin Analysis**: Profit margin percentage visualization
- **Interactive Scatter Plot**: Revenue vs Margin with quadrant coloring

**Technology Used:**
- BCG matrix analysis
- Pandas aggregation
- Plotly scatter plots with custom styling
- Dynamic product name resolution

---

### 🚨 Customer Churn Prediction

![Customer Churn Risk Distribution](project%20Screenshots/Customer_Churn_Risk_Distribution.png)

![Customer Churn High Risk Details](project%20Screenshots/Customer_Churn_High_Risk_Details.png)

**Key Features:**
- **Churn Risk Distribution**: 
  - High Risk (>70% churn probability)
  - Medium Risk (40-70%)
  - Low Risk (<40%)
- **Risk Segmentation**: Bar chart showing customer count by risk tier
- **High-Risk Customer Details**: Table with customer profiles, segments, and risk scores
- **Churn Features**: 
  - Days since last order
  - Customer tenure
  - Order frequency
  - Average order value
- **Temporal Train/Test Split**: Proper time-based model evaluation

**Technology Used:**
- XGBoost for churn prediction
- Temporal cross-validation
- Feature engineering for churn indicators
- Pandera schema validation

---

### 📈 Demand Forecasting

![Demand Forecasting Model Comparison](project%20Screenshots/Demand_Forecasting_Model_Comparison.png)

![Demand Forecasting Moving Average Metrics](project%20Screenshots/Demand_Forecasting_Moving_Average_Metrics.png)

**Key Features:**
- **Model Comparison**:
  - Moving Average (Best performer: MAE 650.13)
  - Exponential Smoothing
  - Seasonal Naive
  - ARIMA
  - ML Ensemble
- **Performance Metrics**: MAE, RMSE, MAPE with cross-validation
- **Forecast Horizon**: Adjustable forecast period (7-90 days)
- **Confidence Intervals**: Upper and lower bounds with 80/90/95% confidence
- **Historical vs Forecast**: Line chart showing historical data and predictions
- **Range Selector**: 7D, 30D, 90D zoom controls
- **Neon-Themed Visualization**: Cyan historical, magenta forecast, orange confidence intervals

**Technology Used:**
- Time series cross-validation
- Multiple forecasting algorithms
- Plotly advanced styling
- Custom neon color scheme

---

### 🚨 Anomaly Detection

![Anomaly Detection Revenue Chart](project%20Screenshots/Anomaly_Detection_Revenue_Chart.png)

![Anomaly Detection Detected Table](project%20Screenshots/Anomaly_Detection_Detected_Table.png)

**Key Features:**
- **Revenue Anomaly Detection**: Z-score based outlier detection
- **Threshold Configuration**: Adjustable Z-score threshold (2.0-4.0)
- **Lookback Period**: Customizable analysis window (30-90 days)
- **Visual Anomaly Markers**: Red dots highlighting detected anomalies
- **Detailed Anomaly Table**: Date, revenue, Z-score, and severity
- **Real-time Alerting**: Automatic flagging of unusual patterns

**Technology Used:**
- Statistical Z-score analysis
- Pandas rolling calculations
- Plotly anomaly visualization
- Configurable detection parameters

---

### 🎯 Decision Center

![Decision Center Business Alerts](project%20Screenshots/Decision_Center_Business_Alerts.png)

![Decision Center Smart Recommendations](project%20Screenshots/Decision_Center_Smart_Recommendations.png)

![Decision Center What If Analysis](project%20Screenshots/Decision_Center_What_If_Analysis.png)

**Key Features:**
- **Business Alerts**:
  - Stock-out warnings
  - Churn risk alerts
  - Revenue anomalies
  - Marketing budget alerts
- **Smart Recommendations**:
  - Inventory reorder suggestions
  - Customer retention strategies
  - Marketing campaign optimizations
  - Pricing adjustments
- **What-If Analysis**:
  - Scenario modeling
  - Impact simulation
  - Sensitivity analysis
- **Priority-Based Actions**: Urgent, high, medium, low priority categorization

**Technology Used:**
- Rule-based decision engine
- Scenario simulation
- Priority scoring algorithms
- Real-time alert generation

---

### 🌍 Regional Performance

![Regional Performance Top States](project%20Screenshots/Regional_Performance_Top_States.png)

**Key Features:**
- **Geographic Analysis**: Revenue by state/region
- **Top Performing Regions**: Ranked list of best-performing areas
- **Regional Metrics**: Revenue, orders, customers per region
- **Growth Rates**: Year-over-year regional growth
- **Heat Map Visualization**: Geographic performance mapping

**Technology Used:**
- Geographic data aggregation
- Regional growth calculations
- Plotly geographic visualizations

---

### 📊 Revenue Trends & Categories

![Revenue Trend and Categories](project%20Screenshots/Revenue-trend-and-categories.png)

**Key Features:**
- **Revenue Trend Analysis**: Monthly/quarterly revenue patterns
- **Category Performance**: Revenue by product category
- **Growth Comparison**: Category-wise growth rates
- **Trend Lines**: Moving averages and trend projections
- **Category Breakdown**: Detailed category metrics

**Technology Used:**
- Time series aggregation
- Category-based analysis
- Trend line calculations
- Multi-series plotting

---

### 👥 Customer Acquisition Trends

![New VS Returning Customers Trend](project%20Screenshots/New-VS-Returning-Customers-Trend.png)

**Key Features:**
- **New vs Returning Customers**: Comparison of customer types over time
- **Acquisition Rate**: New customer acquisition trends
- **Retention Rate**: Returning customer patterns
- **Customer Growth**: Overall customer base growth
- **Cohort Analysis**: Customer behavior by acquisition period

**Technology Used:**
- Customer classification algorithms
- Cohort analysis
- Trend comparison charts
- Growth rate calculations

---

## 🛠️ Technology Stack

### Core
- **Python 3.11+**: Primary programming language
- **PostgreSQL 16**: Data warehouse (optional)
- **SQL**: Advanced analytics and transformations

### Data Processing
- **Pandas**: Data manipulation and analysis
- **Polars**: High-performance data processing
- **NumPy**: Numerical computing
- **SciPy**: Statistical analysis

### Machine Learning
- **Scikit-learn**: ML algorithms and utilities
- **XGBoost**: Gradient boosting for churn prediction
- **Prophet**: Time series forecasting
- **Lifetimes**: BG/NBD + Gamma-Gamma CLV modeling
- **PyOD**: Outlier detection

### Validation
- **Pandera**: Data validation and schema enforcement
- **Great Expectations**: Data quality testing

### Visualization
- **Plotly**: Interactive visualizations with advanced styling
- **Streamlit**: Web application with neon-themed UI
- **Plotly Graph Objects**: Advanced chart customization

### API & Infrastructure
- **FastAPI**: REST API (optional)
- **Docker**: Containerization (optional)
- **Docker Compose**: Multi-container orchestration (optional)

## 📁 Project Structure

```
ecommerce-intelligence/
│
├── data/
│   ├── raw/              # Original synthetic data files
│   ├── staging/          # Cleaned data
│   └── processed/        # Transformed analytics datasets
│
├── src/
│   ├── ingestion/        # Data generation and loading
│   ├── cleaning/         # Data cleaning and profiling
│   ├── validation/       # Data integrity validation
│   ├── transformation/   # Feature engineering
│   ├── analytics/        # Business analytics modules
│   ├── forecasting/      # Demand forecasting with time series CV
│   ├── ml/               # Machine learning models (churn, anomaly)
│   ├── clv/              # BG/NBD + Gamma-Gamma CLV prediction
│   ├── recommendation/   # Product recommendation system
│   └── decision_engine/  # Automated recommendations
│
├── sql/
│   ├── staging/          # Database schema and views
│   ├── transformations/  # Advanced transformation views
│   ├── analytics/        # Analytics SQL views
│   └── kpis/             # KPI calculations
│
├── app/
│   └── streamlit/        # Streamlit application with neon UI
│       ├── main.py       # Main application entry point
│       ├── pages/        # Modular page components
│       ├── styles.py     # Custom CSS styling
│       └── utils.py      # Utility functions
│
├── api/
│   └── fastapi/          # REST API (optional)
│
├── models/               # Trained ML models
├── tests/                # Test suite
├── notebooks/            # Jupyter notebooks
├── docs/                 # Documentation
├── config/               # Configuration files
├── scripts/              # Utility scripts
│   └── generate_dataset.py  # Synthetic data generation
│
├── project Screenshots/  # Application screenshots
├── requirements.txt      # Python dependencies
├── docker-compose.yml   # Container orchestration (optional)
├── .env.example         # Environment variables template
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 16+ (optional)
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "E-Commerce Intelligence & Decision Analytics Platform"
```

2. **Create virtual environment**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate synthetic data**
```bash
python scripts/generate_dataset.py --scale 1.0
```

This will generate:
- 100,000 customers
- 5,000 products
- 200,000 orders
- 426,912 order items
- Complete relational dataset

5. **Launch Streamlit app**
```bash
python -m streamlit run app/streamlit/main.py
```

The application will be available at: **http://localhost:8501**

### Docker Setup (Optional)

1. **Start all services**
```bash
docker-compose up -d
```

2. **Access services**
- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000
- PostgreSQL: localhost:5432
- Adminer: http://localhost:8081

3. **Stop services**
```bash
docker-compose down
```

## 📊 Usage Examples

### Running Analytics

```python
from src.analytics.customer_analytics import CustomerIntelligence
from src.analytics.product_analytics import ProductIntelligence
from src.analytics.marketing_analytics import MarketingAnalyzer

# Customer Intelligence
ci = CustomerIntelligence(data_dir="data/raw")
customer_360 = ci.build_customer_360()
customer_360.to_csv("data/processed/customer_360.csv", index=False)

# Product Intelligence
pi = ProductIntelligence(data_dir="data/raw")
product_analytics = pi.run_all_product_analytics()

# Marketing Analytics
ma = MarketingAnalyzer(data_dir="data/raw")
marketing_analytics = ma.run_all_marketing_analytics()
```

### Running ML Models

```python
from src.ml.churn_predictor import ChurnPredictor
from src.clv.clv_predictor import CLVPredictor
from src.forecasting.demand_forecaster import DemandForecaster
from src.recommendation.recommender import ProductRecommender

# Churn Prediction (with temporal split)
cp = ChurnPredictor()
churn_model = cp.train_model(use_temporal_split=True)
predictions = cp.predict_churn_risk()

# CLV Prediction (BG/NBD + Gamma-Gamma)
clv = CLVPredictor()
clv_results = clv.run_all(data_dir="data/raw", processed_dir="data/processed")

# Demand Forecasting (with time series cross-validation)
df = DemandForecaster()
forecast = df.run_all(horizon=30, save=True)

# Product Recommendations
pr = ProductRecommender()
pr.load_all(data_dir="data/raw")
recommendations = pr.hybrid_recommendations(customer_id=12345, n_recommendations=5)
```

### Using Decision Engine

```python
from src.decision_engine.decision_center import DecisionCenter

dc = DecisionCenter()
recommendations = dc.generate_recommendations()
alerts = dc.get_active_alerts()
```

## 🎨 UI/UX Features

### Neon-Themed Design
- **Dark Theme**: Professional dark background with neon accents
- **Color Palette**: Cyan (#00ffff), Magenta (#ff00ff), Green (#00ff00), Orange (#ff6600)
- **Animations**: Smooth transitions, floating effects, pulse animations
- **Glassmorphism**: Frosted glass effect on cards and containers
- **Responsive Design**: Adapts to different screen sizes

### Interactive Elements
- **Hover Effects**: Neon glow on hover
- **Smooth Scrolling**: Animated page transitions
- **Custom Scrollbars**: Styled scrollbars matching theme
- **Icon Animations**: Pulsing navigation icons
- **Chart Interactions**: Zoom, pan, hover tooltips

### Advanced Chart Features
- **Range Selectors**: Time period selection buttons
- **Confidence Intervals**: Shaded uncertainty regions
- **Vertical Markers**: Key event indicators
- **Custom Hover Templates**: Formatted tooltips
- **Neon Grid Lines**: Styled axis grids

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📈 Key Metrics

The platform tracks these business KPIs:

### Executive Metrics
- **Revenue**: Calculated from actual order data
- **Profit**: Gross profit with configurable margin
- **Orders**: Total order count with status breakdown
- **Customers**: Unique customer count with segmentation
- **AOV**: Average order value
- **Return Rate**: Return percentage based on order status

### Customer Metrics
- **RFM Segments**: 7 customer segments based on behavior
- **CLV**: BG/NBD + Gamma-Gamma probabilistic lifetime value modeling
- **Churn Rate**: 90-day churn prediction with temporal split
- **Retention**: Cohort retention matrices

### Product Metrics
- **Product Matrix**: 4-quadrant analysis (Stars/Volume/Remove/Premium)
- **Inventory**: Stock-out prediction, reorder recommendations
- **Lifecycle**: Product stage classification

### ML Model Metrics
- **Churn Model**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Forecasting**: MAE, RMSE, MAPE with time series cross-validation
- **CLV Model**: BG/NBD parameters, Gamma-Gamma parameters

## 🔧 Configuration

Key configuration options in `.env`:

```env
# Database (optional)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=ecommerce_user
POSTGRES_PASSWORD=ecommerce_pass
POSTGRES_DB=ecommerce_warehouse

# Data Generation
NUM_CUSTOMERS=100000
NUM_PRODUCTS=5000
NUM_ORDERS=200000
DATA_START_DATE=2022-01-01
DATA_END_DATE=2024-12-31
ANALYSIS_AS_OF_DATE=2024-12-31

# Application
STREAMLIT_SERVER_PORT=8501
FASTAPI_PORT=8000
```

## 📚 Dependencies

### Core Dependencies
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
```

### Data Processing
```
polars>=0.19.0
scipy>=1.11.0
```

### Machine Learning
```
scikit-learn>=1.3.0
xgboost>=2.0.0
prophet>=1.1.4
lifetimes>=0.11.1
pyod>=1.0.0
```

### Validation
```
pandera>=0.17.0
great-expectations>=0.17.0
```

### API & Infrastructure
```
fastapi>=0.103.0
uvicorn>=0.23.0
docker>=6.1.0
docker-compose>=1.29.0
```

### Utilities
```
python-dotenv>=1.0.0
pyyaml>=6.0.0
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Data Model Documentation](docs/data_model.md)
- [Contributing Guide](docs/CONTRIBUTING.md)

## 🤝 Contributing

Contributions are welcome! Please read the contributing guide before submitting PRs.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with modern data engineering best practices
- Inspired by production analytics platforms
- Uses industry-standard tools and frameworks
- Neon UI design inspired by modern dashboard aesthetics

## 📞 Contact

For questions or support, please open an issue in the repository.

---

**Version**: 2.0.0  
**Last Updated**: September 2026  
**Status**: Production-Ready with Neon UI
