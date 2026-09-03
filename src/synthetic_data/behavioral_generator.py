"""
Behavioral Synthetic Data Generator Module
Implements synthetic data generation with behavioral simulation for testing and development.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from faker import Faker
from config.logging_config import get_logger

logger = get_logger(__name__)


class BehavioralDataGenerator:
    """
    Behavioral synthetic data generator.
    
    Features:
    - Customer behavior simulation (purchase patterns, churn, engagement)
    - Product interaction simulation
    - Time-series event generation
    - Realistic behavioral patterns
    - Configurable behavior parameters
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize behavioral data generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        self.faker = Faker()
        Faker.seed(random_seed)
        
        logger.info("Behavioral data generator initialized")
    
    def generate_customers(
        self,
        n_customers: int = 1000,
        start_date: str = '2023-01-01',
        end_date: str = '2024-12-31'
    ) -> pd.DataFrame:
        """
        Generate synthetic customer data with behavioral attributes.
        
        Args:
            n_customers: Number of customers to generate
            start_date: Start date for customer registration
            end_date: End date for customer registration
        
        Returns:
            DataFrame with customer data
        """
        logger.info(f"Generating {n_customers} synthetic customers...")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        customers = []
        
        for i in range(n_customers):
            # Registration date (weighted towards earlier dates)
            days_since_start = np.random.exponential(scale=365)
            registration_date = start + timedelta(days=min(days_since_start, (end - start).days))
            
            # Behavioral attributes
            purchase_frequency = np.random.gamma(shape=2, scale=30)  # Average days between purchases
            avg_basket_value = np.random.lognormal(mean=3, sigma=0.5)  # Log-normal distribution
            price_sensitivity = np.random.beta(a=2, b=2)  # 0-1, higher = more sensitive
            discount_dependency = np.random.beta(a=1.5, b=3)  # 0-1, higher = more dependent
            return_propensity = np.random.beta(a=1, b=4)  # 0-1, higher = more returns
            
            # Churn risk (based on behavioral attributes)
            churn_risk = 0.3 * (1 - price_sensitivity) + 0.4 * discount_dependency + 0.3 * return_propensity
            churn_risk = min(churn_risk, 0.9)  # Cap at 90%
            
            customers.append({
                'customer_id': f'CUST_{i:06d}',
                'registration_date': registration_date,
                'email': self.faker.email(),
                'city': self.faker.city(),
                'country': self.faker.country(),
                'age': np.random.randint(18, 80),
                'gender': np.random.choice(['M', 'F', 'Other'], p=[0.45, 0.45, 0.1]),
                'purchase_frequency_days': purchase_frequency,
                'avg_basket_value': avg_basket_value,
                'price_sensitivity': price_sensitivity,
                'discount_dependency': discount_dependency,
                'return_propensity': return_propensity,
                'churn_risk': churn_risk,
                'is_active': np.random.random() > churn_risk
            })
        
        customers_df = pd.DataFrame(customers)
        
        logger.info(f"Generated {len(customers_df)} customers")
        
        return customers_df
    
    def generate_products(
        self,
        n_products: int = 500,
        n_categories: int = 20
    ) -> pd.DataFrame:
        """
        Generate synthetic product data.
        
        Args:
            n_products: Number of products to generate
            n_categories: Number of product categories
        
        Returns:
            DataFrame with product data
        """
        logger.info(f"Generating {n_products} synthetic products...")
        
        categories = [f'CAT_{i}' for i in range(n_categories)]
        brands = [f'Brand_{i}' for i in range(50)]
        
        products = []
        
        for i in range(n_products):
            category = np.random.choice(categories)
            base_price = np.random.lognormal(mean=3.5, sigma=0.8)
            
            # Price varies by category
            category_multiplier = np.random.uniform(0.5, 2.0)
            selling_price = base_price * category_multiplier
            
            # Product popularity (affects demand)
            popularity = np.random.beta(a=2, b=2)
            
            # Return rate (varies by price - more expensive items have lower return rate)
            return_rate = max(0.05, 0.2 - 0.1 * np.log10(selling_price))
            
            products.append({
                'product_id': f'PROD_{i:06d}',
                'product_name': f'Product {i}',
                'category_id': category,
                'brand_name': np.random.choice(brands),
                'selling_price': selling_price,
                'cost_price': selling_price * np.random.uniform(0.3, 0.7),
                'popularity': popularity,
                'return_rate': return_rate,
                'stock_quantity': np.random.randint(0, 1000)
            })
        
        products_df = pd.DataFrame(products)
        
        logger.info(f"Generated {len(products_df)} products")
        
        return products_df
    
    def generate_orders(
        self,
        customers_df: pd.DataFrame,
        products_df: pd.DataFrame,
        n_orders: int = 10000,
        start_date: str = '2023-01-01',
        end_date: str = '2024-12-31'
    ) -> pd.DataFrame:
        """
        Generate synthetic order data with behavioral patterns.
        
        Args:
            customers_df: Customer data
            products_df: Product data
            n_orders: Number of orders to generate
            start_date: Start date for orders
            end_date: End date for orders
        
        Returns:
            DataFrame with order data
        """
        logger.info(f"Generating {n_orders} synthetic orders...")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        orders = []
        order_items = []
        
        for i in range(n_orders):
            # Select customer (weighted by activity)
            active_customers = customers_df[customers_df['is_active']]
            customer = active_customers.sample(1).iloc[0]
            
            # Calculate order date based on purchase frequency
            days_since_registration = (start - customer['registration_date']).days
            if days_since_registration < 0:
                days_since_registration = 0
            
            # Generate order date
            order_date = start + timedelta(
                days=np.random.randint(0, (end - start).days)
            )
            
            # Calculate basket size based on customer behavior
            n_items = np.random.poisson(lam=3) + 1
            n_items = min(n_items, 10)  # Cap at 10 items
            
            # Select products (weighted by popularity)
            selected_products = products_df.sample(
                n=n_items,
                weights=products_df['popularity'],
                replace=True
            )
            
            # Calculate discounts based on customer's discount dependency
            discount_pct = 0
            if customer['discount_dependency'] > 0.5:
                discount_pct = np.random.uniform(0.05, 0.3)
            
            # Calculate order total
            order_total = 0
            for _, product in selected_products.iterrows():
                quantity = np.random.randint(1, 4)
                price = product['selling_price'] * (1 - discount_pct)
                item_total = price * quantity
                order_total += item_total
                
                # Determine if item is returned (based on customer's return propensity and product's return rate)
                is_returned = np.random.random() < (customer['return_propensity'] * product['return_rate'])
                
                order_items.append({
                    'order_id': f'ORD_{i:06d}',
                    'product_id': product['product_id'],
                    'quantity': quantity,
                    'unit_price': product['selling_price'],
                    'discount_pct': discount_pct,
                    'item_total': item_total,
                    'is_returned': is_returned
                })
            
            orders.append({
                'order_id': f'ORD_{i:06d}',
                'customer_id': customer['customer_id'],
                'order_date': order_date,
                'n_items': n_items,
                'order_total': order_total,
                'discount_pct': discount_pct,
                'payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal', 'cash']),
                'order_status': np.random.choice(['completed', 'pending', 'cancelled'], p=[0.85, 0.1, 0.05])
            })
        
        orders_df = pd.DataFrame(orders)
        order_items_df = pd.DataFrame(order_items)
        
        logger.info(f"Generated {len(orders_df)} orders with {len(order_items_df)} items")
        
        return orders_df, order_items_df
    
    def generate_reviews(
        self,
        orders_df: pd.DataFrame,
        products_df: pd.DataFrame,
        review_rate: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate synthetic product reviews with sentiment.
        
        Args:
            orders_df: Order data
            products_df: Product data
            review_rate: Probability of a customer leaving a review
        
        Returns:
            DataFrame with review data
        """
        logger.info("Generating synthetic reviews...")
        
        reviews = []
        
        for _, order in orders_df.iterrows():
            if np.random.random() > review_rate:
                continue
            
            # Rating distribution (skewed towards positive)
            rating = np.random.choice([5, 4, 3, 2, 1], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            
            # Review text templates
            positive_templates = [
                "Great product, very satisfied!",
                "Excellent quality, would buy again.",
                "Fast shipping and good condition.",
                "Love this product, highly recommend.",
                "Perfect for my needs."
            ]
            
            negative_templates = [
                "Not as described, disappointed.",
                "Poor quality, would not recommend.",
                "Arrived damaged, very unhappy.",
                "Overpriced for what you get.",
                "Not worth the money."
            ]
            
            neutral_templates = [
                "Average product, nothing special.",
                "It's okay, could be better.",
                "Decent quality for the price.",
                "Met expectations but didn't exceed them."
            ]
            
            if rating >= 4:
                text = np.random.choice(positive_templates)
            elif rating <= 2:
                text = np.random.choice(negative_templates)
            else:
                text = np.random.choice(neutral_templates)
            
            reviews.append({
                'review_id': f'REV_{len(reviews):06d}',
                'order_id': order['order_id'],
                'customer_id': order['customer_id'],
                'rating': rating,
                'review_text': text,
                'review_date': order['order_date'] + timedelta(days=np.random.randint(1, 30))
            })
        
        reviews_df = pd.DataFrame(reviews)
        
        logger.info(f"Generated {len(reviews_df)} reviews")
        
        return reviews_df
    
    def generate_marketing_events(
        self,
        customers_df: pd.DataFrame,
        start_date: str = '2023-01-01',
        end_date: str = '2024-12-31',
        n_campaigns: int = 20
    ) -> pd.DataFrame:
        """
        Generate synthetic marketing campaign events.
        
        Args:
            customers_df: Customer data
            start_date: Start date for campaigns
            end_date: End date for campaigns
            n_campaigns: Number of campaigns
        
        Returns:
            DataFrame with marketing events
        """
        logger.info(f"Generating {n_campaigns} marketing campaigns...")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        campaign_types = ['email', 'social_media', 'display_ads', 'search_ads', 'direct_mail']
        
        campaigns = []
        campaign_events = []
        
        for i in range(n_campaigns):
            # Campaign dates
            campaign_start = start + timedelta(
                days=np.random.randint(0, (end - start).days - 30)
            )
            campaign_end = campaign_start + timedelta(days=np.random.randint(7, 30))
            
            # Campaign attributes
            campaign_type = np.random.choice(campaign_types)
            budget = np.random.lognormal(mean=10, sigma=0.5)
            
            campaigns.append({
                'campaign_id': f'CAMP_{i:06d}',
                'campaign_name': f'Campaign {i}',
                'campaign_type': campaign_type,
                'start_date': campaign_start,
                'end_date': campaign_end,
                'budget': budget,
                'target_audience': np.random.choice(['all', 'high_value', 'at_risk', 'new'])
            })
        
        campaigns_df = pd.DataFrame(campaigns)
        
        # Generate campaign events for customers
        for _, campaign in campaigns_df.iterrows():
            # Select customers for campaign
            target_customers = customers_df.sample(
                n=int(len(customers_df) * np.random.uniform(0.1, 0.5))
            )
            
            for _, customer in target_customers.iterrows():
                # Event date during campaign
                event_date = campaign['start_date'] + timedelta(
                    days=np.random.randint(0, (campaign['end_date'] - campaign['start_date']).days)
                )
                
                # Engagement (click, open, convert)
                engagement = np.random.choice(['sent', 'opened', 'clicked', 'converted'], 
                                              p=[0.4, 0.3, 0.2, 0.1])
                
                campaign_events.append({
                    'campaign_id': campaign['campaign_id'],
                    'customer_id': customer['customer_id'],
                    'event_date': event_date,
                    'engagement_type': engagement
                })
        
        campaign_events_df = pd.DataFrame(campaign_events)
        
        logger.info(f"Generated {len(campaigns_df)} campaigns with {len(campaign_events_df)} events")
        
        return campaigns_df, campaign_events_df


def run_behavioral_simulation_pipeline(
    n_customers: int = 1000,
    n_products: int = 500,
    n_orders: int = 10000,
    start_date: str = '2023-01-01',
    end_date: str = '2024-12-31'
) -> Dict:
    """
    Convenience function to run complete behavioral simulation pipeline.
    
    Args:
        n_customers: Number of customers
        n_products: Number of products
        n_orders: Number of orders
        start_date: Start date
        end_date: End date
    
    Returns:
        Dictionary with all generated data
    """
    generator = BehavioralDataGenerator()
    
    # Generate customers
    customers_df = generator.generate_customers(n_customers, start_date, end_date)
    
    # Generate products
    products_df = generator.generate_products(n_products)
    
    # Generate orders
    orders_df, order_items_df = generator.generate_orders(
        customers_df, products_df, n_orders, start_date, end_date
    )
    
    # Generate reviews
    reviews_df = generator.generate_reviews(orders_df, products_df)
    
    # Generate marketing events
    campaigns_df, campaign_events_df = generator.generate_marketing_events(
        customers_df, start_date, end_date
    )
    
    results = {
        'customers': customers_df,
        'products': products_df,
        'orders': orders_df,
        'order_items': order_items_df,
        'reviews': reviews_df,
        'campaigns': campaigns_df,
        'campaign_events': campaign_events_df
    }
    
    return results
