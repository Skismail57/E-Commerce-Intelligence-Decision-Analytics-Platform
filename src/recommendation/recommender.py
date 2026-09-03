"""
Product Recommendation System
Implements collaborative filtering and content-based recommendations for e-commerce.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

from config.logging_config import get_logger

logger = get_logger(__name__)


class ProductRecommender:
    """
    Product recommendation engine using multiple strategies:
    - Collaborative filtering (user-based and item-based)
    - Content-based filtering (category, brand similarity)
    - Popularity-based (for cold start)
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self._dfs: Dict[str, pd.DataFrame] = {}
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.item_similarity: Optional[pd.DataFrame] = None
        self.popular_products: Optional[pd.DataFrame] = None
    
    def _read_csv(self, table: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
        directory = data_dir or self.data_dir
        if directory is None:
            raise ValueError("data_dir must be provided")
        path = Path(directory) / f"{table}.csv"
        df = pd.read_csv(path, low_memory=False)
        return df
    
    def load_all(self, data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
        """Load all required tables."""
        directory = data_dir or self.data_dir
        tables = [
            "customers", "products", "categories", "orders", "order_items",
        ]
        for t in tables:
            try:
                self._dfs[t] = self._read_csv(t, directory)
            except FileNotFoundError:
                logger.warning(f"Table {t} not found, skipping")
        logger.info(f"Loaded {len(self._dfs)} tables")
        return self._dfs
    
    def build_user_item_matrix(self) -> pd.DataFrame:
        """
        Build user-item interaction matrix.
        Rows: customers, Columns: products, Values: purchase count or total spend.
        """
        if "orders" not in self._dfs or "order_items" not in self._dfs:
            raise ValueError("Orders and order_items data required")
        
        orders_df = self._dfs["orders"]
        order_items_df = self._dfs["order_items"]
        
        # Merge orders with order items
        user_items = order_items_df.merge(
            orders_df[["order_id", "customer_id"]], on="order_id", how="left"
        )
        
        # Create user-item matrix (using quantity as interaction strength)
        self.user_item_matrix = user_items.groupby(
            ["customer_id", "product_id"]
        )["quantity"].sum().unstack(fill_value=0)
        
        logger.info(f"Built user-item matrix: {self.user_item_matrix.shape}")
        return self.user_item_matrix
    
    def compute_item_similarity(self, method: str = "cosine") -> pd.DataFrame:
        """
        Compute item-item similarity matrix.
        
        Args:
            method: Similarity metric ('cosine', 'jaccard')
        """
        if self.user_item_matrix is None:
            self.build_user_item_matrix()
        
        # Transpose to get item-user matrix
        item_user = self.user_item_matrix.T
        
        # Memory-efficient handling for large matrices
        if item_user.shape[1] > 10000:
            logger.warning(f"Large user-item matrix ({item_user.shape}), using top 10000 users for similarity")
            item_user = item_user.iloc[:, :10000]
        
        if method == "cosine":
            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            try:
                similarity = cosine_similarity(item_user)
            except MemoryError:
                logger.warning("Memory error computing full similarity, using batch approach")
                # Fallback: compute similarity for top items only
                top_items = item_user.head(500)
                similarity = cosine_similarity(top_items)
                item_user = top_items
        elif method == "jaccard":
            # Jaccard similarity
            from sklearn.metrics import jaccard_score
            # Convert to binary
            item_user_binary = (item_user > 0).astype(int)
            similarity = np.zeros((len(item_user), len(item_user)))
            for i in range(len(item_user)):
                for j in range(len(item_user)):
                    similarity[i, j] = jaccard_score(
                        item_user_binary.iloc[i], item_user_binary.iloc[j]
                    )
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        self.item_similarity = pd.DataFrame(
            similarity,
            index=item_user.index,
            columns=item_user.index
        )
        
        logger.info(f"Computed item similarity matrix: {self.item_similarity.shape}")
        return self.item_similarity
    
    def compute_popular_products(self, top_n: int = 100) -> pd.DataFrame:
        """
        Compute popular products based on purchase frequency and revenue.
        """
        if "order_items" not in self._dfs:
            raise ValueError("Order items data required")
        
        order_items_df = self._dfs["order_items"]
        
        product_stats = order_items_df.groupby("product_id").agg({
            "quantity": "sum",
            "line_total": "sum",
            "order_id": "nunique"
        }).reset_index()
        product_stats.columns = ["product_id", "total_quantity", "total_revenue", "num_orders"]
        
        # Sort by combined score (quantity + orders)
        product_stats["popularity_score"] = (
            product_stats["total_quantity"] * 0.5 + 
            product_stats["num_orders"] * 0.5
        )
        product_stats = product_stats.sort_values("popularity_score", ascending=False)
        
        self.popular_products = product_stats.head(top_n)
        logger.info(f"Computed top {top_n} popular products")
        return self.popular_products
    
    def collaborative_filtering_recommendations(
        self,
        customer_id: int,
        n_recommendations: int = 5,
        method: str = "item_based"
    ) -> List[Tuple[int, float]]:
        """
        Generate collaborative filtering recommendations.
        
        Args:
            customer_id: Target customer ID
            n_recommendations: Number of recommendations to return
            method: 'item_based' or 'user_based'
        
        Returns:
            List of (product_id, score) tuples
        """
        if self.user_item_matrix is None:
            self.build_user_item_matrix()
        
        if customer_id not in self.user_item_matrix.index:
            logger.warning(f"Customer {customer_id} not in user-item matrix")
            return []
        
        if method == "item_based":
            if self.item_similarity is None:
                self.compute_item_similarity()
            
            # Get products the customer has purchased
            customer_purchases = self.user_item_matrix.loc[customer_id]
            purchased_products = customer_purchases[customer_purchases > 0].index.tolist()
            
            if not purchased_products:
                return []
            
            # Compute recommendation scores based on similar items
            recommendation_scores = defaultdict(float)
            
            for product in purchased_products:
                # Get similar items
                if product in self.item_similarity.index:
                    similar_items = self.item_similarity.loc[product].sort_values(ascending=False)
                    
                    # Skip already purchased items
                    for similar_item, similarity in similar_items.items():
                        if similar_item not in purchased_products and similarity > 0:
                            recommendation_scores[similar_item] += similarity * customer_purchases[product]
            
            # Sort and return top N
            recommendations = sorted(
                recommendation_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            return recommendations
        
        elif method == "user_based":
            # User-based collaborative filtering
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Compute user similarity
            user_similarity = cosine_similarity(self.user_item_matrix)
            user_similarity_df = pd.DataFrame(
                user_similarity,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index
            )
            
            # Get similar users
            if customer_id not in user_similarity_df.index:
                return []
            
            similar_users = user_similarity_df.loc[customer_id].sort_values(ascending=False)
            similar_users = similar_users[similar_users > 0][1:20]  # Top 20 similar users, excluding self
            
            # Get products purchased by similar users
            recommendation_scores = defaultdict(float)
            customer_purchases = set(self.user_item_matrix.loc[customer_id][
                self.user_item_matrix.loc[customer_id] > 0
            ].index)
            
            for similar_user_id, similarity in similar_users.items():
                user_purchases = self.user_item_matrix.loc[similar_user_id]
                for product_id, quantity in user_purchases[user_purchases > 0].items():
                    if product_id not in customer_purchases:
                        recommendation_scores[product_id] += similarity * quantity
            
            recommendations = sorted(
                recommendation_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            return recommendations
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def content_based_recommendations(
        self,
        customer_id: int,
        n_recommendations: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Generate content-based recommendations based on category and brand preferences.
        
        Args:
            customer_id: Target customer ID
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of (product_id, score) tuples
        """
        if "orders" not in self._dfs or "order_items" not in self._dfs or "products" not in self._dfs:
            raise ValueError("Orders, order_items, and products data required")
        
        orders_df = self._dfs["orders"]
        order_items_df = self._dfs["order_items"]
        products_df = self._dfs["products"]
        
        # Get customer's purchase history
        customer_orders = orders_df[orders_df["customer_id"] == customer_id]
        if len(customer_orders) == 0:
            return []
        
        customer_order_items = order_items_df[
            order_items_df["order_id"].isin(customer_orders["order_id"])
        ]
        
        # Merge with products to get category and brand info
        customer_products = customer_order_items.merge(
            products_df[["product_id", "category_id", "brand_name"]], 
            on="product_id", 
            how="left"
        )
        
        # Calculate category and brand preferences
        category_preference = customer_products.groupby("category_id")["quantity"].sum()
        brand_preference = customer_products.groupby("brand_name")["quantity"].sum()
        
        # Get purchased product IDs
        purchased_product_ids = set(customer_order_items["product_id"].unique())
        
        # Score products based on category and brand similarity
        recommendation_scores = defaultdict(float)
        
        for _, product in products_df.iterrows():
            product_id = product["product_id"]
            if product_id in purchased_product_ids:
                continue
            
            score = 0.0
            
            # Category score
            if product["category_id"] in category_preference.index:
                score += category_preference[product["category_id"]] * 2.0
            
            # Brand score
            if product["brand_name"] in brand_preference.index:
                score += brand_preference[product["brand_name"]] * 1.0
            
            if score > 0:
                recommendation_scores[product_id] = score
        
        recommendations = sorted(
            recommendation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        return recommendations
    
    def popularity_based_recommendations(
        self,
        n_recommendations: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Generate popularity-based recommendations (for cold start).
        
        Args:
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of (product_id, score) tuples
        """
        if self.popular_products is None:
            self.compute_popular_products()
        
        recommendations = [
            (row["product_id"], row["popularity_score"])
            for _, row in self.popular_products.head(n_recommendations).iterrows()
        ]
        
        return recommendations
    
    def hybrid_recommendations(
        self,
        customer_id: int,
        n_recommendations: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate hybrid recommendations combining multiple strategies.
        
        Args:
            customer_id: Target customer ID
            n_recommendations: Number of recommendations to return
            weights: Strategy weights (default: collaborative=0.5, content=0.3, popularity=0.2)
        
        Returns:
            List of (product_id, score) tuples
        """
        if weights is None:
            weights = {
                "collaborative": 0.5,
                "content": 0.3,
                "popularity": 0.2
            }
        
        # Get recommendations from each strategy
        collaborative_recs = self.collaborative_filtering_recommendations(
            customer_id, n_recommendations * 2
        )
        content_recs = self.content_based_recommendations(
            customer_id, n_recommendations * 2
        )
        popularity_recs = self.popularity_based_recommendations(
            n_recommendations * 2
        )
        
        # Combine scores
        combined_scores = defaultdict(float)
        
        for product_id, score in collaborative_recs:
            combined_scores[product_id] += score * weights["collaborative"]
        
        for product_id, score in content_recs:
            combined_scores[product_id] += score * weights["content"]
        
        for product_id, score in popularity_recs:
            combined_scores[product_id] += score * weights["popularity"]
        
        # Sort and return top N
        recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        return recommendations
    
    def get_recommendations_with_details(
        self,
        customer_id: int,
        n_recommendations: int = 5,
        strategy: str = "hybrid"
    ) -> pd.DataFrame:
        """
        Get recommendations with product details.
        
        Args:
            customer_id: Target customer ID
            n_recommendations: Number of recommendations
            strategy: 'collaborative', 'content', 'popularity', or 'hybrid'
        
        Returns:
            DataFrame with product details and recommendation scores
        """
        if strategy == "collaborative":
            recommendations = self.collaborative_filtering_recommendations(
                customer_id, n_recommendations
            )
        elif strategy == "content":
            recommendations = self.content_based_recommendations(
                customer_id, n_recommendations
            )
        elif strategy == "popularity":
            recommendations = self.popularity_based_recommendations(
                n_recommendations
            )
        elif strategy == "hybrid":
            recommendations = self.hybrid_recommendations(
                customer_id, n_recommendations
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if not recommendations:
            return pd.DataFrame()
        
        product_ids = [pid for pid, _ in recommendations]
        scores = {pid: score for pid, score in recommendations}
        
        if "products" not in self._dfs:
            return pd.DataFrame({"product_id": product_ids, "score": [scores[pid] for pid in product_ids]})
        
        products_df = self._dfs["products"]
        recommended_products = products_df[products_df["product_id"].isin(product_ids)].copy()
        recommended_products["recommendation_score"] = recommended_products["product_id"].map(scores)
        recommended_products = recommended_products.sort_values("recommendation_score", ascending=False)
        
        return recommended_products
    
    def run_all(
        self,
        data_dir: Optional[Path] = None,
        save: bool = True
    ) -> Dict:
        """
        Run the full recommendation pipeline.
        
        Args:
            data_dir: Directory containing data files
            save: Whether to save outputs
        
        Returns:
            Dictionary with pipeline results
        """
        self.data_dir = Path(data_dir) if data_dir else self.data_dir
        self.load_all(self.data_dir)
        
        # Build matrices
        self.build_user_item_matrix()
        self.compute_item_similarity()
        self.compute_popular_products()
        
        outputs = {
            "user_item_matrix": self.user_item_matrix,
            "item_similarity": self.item_similarity,
            "popular_products": self.popular_products
        }
        
        if save:
            output_dir = self.data_dir.parent / "processed"
            output_dir.mkdir(exist_ok=True)
            
            for name, df in outputs.items():
                if df is not None and len(df) > 0:
                    out_path = output_dir / f"recommendation_{name}.csv"
                    df.to_csv(out_path)
                    logger.info(f"Saved {name} -> {out_path}")
        
        return {
            "status": "success",
            "outputs": {k: v.shape if v is not None else None for k, v in outputs.items()}
        }
