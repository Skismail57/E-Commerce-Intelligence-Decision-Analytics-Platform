"""
Graph Analytics Module
Implements customer-product graph analytics for relationship mining and community detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from config.logging_config import get_logger

logger = get_logger(__name__)


class GraphAnalytics:
    """
    Graph analytics engine for customer-product relationship analysis.
    
    Features:
    - Customer-product bipartite graph construction
    - Community detection
    - Centrality analysis
    - Path analysis
    - Recommendation based on graph structure
    """
    
    def __init__(self):
        """Initialize graph analytics engine"""
        self.graph = defaultdict(set)
        self.node_types = {}
        self.node_attributes = {}
    
    def build_customer_product_graph(
        self,
        orders_df: pd.DataFrame,
        customer_col: str = 'customer_id',
        product_col: str = 'product_id'
    ) -> Dict:
        """
        Build customer-product bipartite graph from order data.
        
        Args:
            orders_df: DataFrame with order data
            customer_col: Customer ID column name
            product_col: Product ID column name
        
        Returns:
            Dictionary with graph statistics
        """
        logger.info("Building customer-product graph...")
        
        # Build adjacency lists
        customer_products = defaultdict(set)
        product_customers = defaultdict(set)
        
        for _, row in orders_df.iterrows():
            customer_id = row[customer_col]
            product_id = row[product_col]
            
            customer_products[customer_id].add(product_id)
            product_customers[product_id].add(customer_id)
        
        # Store graph
        self.graph['customer_products'] = customer_products
        self.graph['product_customers'] = product_customers
        
        # Set node types
        for customer_id in customer_products:
            self.node_types[customer_id] = 'customer'
        for product_id in product_customers:
            self.node_types[product_id] = 'product'
        
        # Calculate statistics
        n_customers = len(customer_products)
        n_products = len(product_customers)
        n_edges = sum(len(products) for products in customer_products.values())
        
        # Calculate degree distribution
        customer_degrees = [len(products) for products in customer_products.values()]
        product_degrees = [len(customers) for customers in product_customers.values()]
        
        stats = {
            'n_customers': n_customers,
            'n_products': n_products,
            'n_edges': n_edges,
            'avg_customer_degree': float(np.mean(customer_degrees)) if customer_degrees else 0,
            'avg_product_degree': float(np.mean(product_degrees)) if product_degrees else 0,
            'max_customer_degree': int(max(customer_degrees)) if customer_degrees else 0,
            'max_product_degree': int(max(product_degrees)) if product_degrees else 0
        }
        
        logger.info(f"Graph built: {n_customers} customers, {n_products} products, {n_edges} edges")
        
        return stats
    
    def calculate_centrality(
        self,
        node_type: str = 'customer'
    ) -> Dict:
        """
        Calculate centrality measures for nodes.
        
        Args:
            node_type: Type of nodes ('customer' or 'product')
        
        Returns:
            Dictionary with centrality measures
        """
        logger.info(f"Calculating centrality for {node_type} nodes...")
        
        if node_type == 'customer':
            adjacency = self.graph['customer_products']
        else:
            adjacency = self.graph['product_customers']
        
        centrality = {}
        
        for node, neighbors in adjacency.items():
            # Degree centrality
            degree = len(neighbors)
            
            # Store centrality
            centrality[node] = {
                'degree': degree,
                'degree_centrality': float(degree)
            }
        
        # Normalize degree centrality
        max_degree = max(c['degree'] for c in centrality.values()) if centrality else 1
        for node in centrality:
            centrality[node]['degree_centrality'] = centrality[node]['degree'] / max_degree
        
        logger.info(f"Centrality calculated for {len(centrality)} nodes")
        
        return centrality
    
    def detect_communities(
        self,
        node_type: str = 'customer',
        min_community_size: int = 5
    ) -> List[Dict]:
        """
        Detect communities using simple clustering based on shared neighbors.
        
        Args:
            node_type: Type of nodes ('customer' or 'product')
            min_community_size: Minimum community size
        
        Returns:
            List of communities
        """
        logger.info(f"Detecting communities for {node_type} nodes...")
        
        if node_type == 'customer':
            adjacency = self.graph['customer_products']
            other_adjacency = self.graph['product_customers']
        else:
            adjacency = self.graph['product_customers']
            other_adjacency = self.graph['customer_products']
        
        # Calculate Jaccard similarity between nodes
        communities = []
        visited = set()
        
        for node in adjacency:
            if node in visited:
                continue
            
            # Find similar nodes
            similar_nodes = [node]
            node_neighbors = adjacency[node]
            
            for other_node in adjacency:
                if other_node == node or other_node in visited:
                    continue
                
                other_neighbors = adjacency[other_node]
                
                # Calculate Jaccard similarity
                intersection = len(node_neighbors & other_neighbors)
                union = len(node_neighbors | other_neighbors)
                jaccard = intersection / union if union > 0 else 0
                
                if jaccard > 0.3:  # Similarity threshold
                    similar_nodes.append(other_node)
            
            if len(similar_nodes) >= min_community_size:
                communities.append({
                    'community_id': len(communities),
                    'nodes': similar_nodes,
                    'size': len(similar_nodes),
                    'node_type': node_type
                })
                visited.update(similar_nodes)
            else:
                visited.add(node)
        
        logger.info(f"Detected {len(communities)} communities")
        
        return communities
    
    def find_shortest_paths(
        self,
        source_node: str,
        target_node: str,
        max_length: int = 5
    ) -> List[List]:
        """
        Find shortest paths between nodes in the bipartite graph.
        
        Args:
            source_node: Source node ID
            target_node: Target node ID
            max_length: Maximum path length
        
        Returns:
            List of paths
        """
        logger.info(f"Finding paths from {source_node} to {target_node}...")
        
        paths = []
        
        # BFS for shortest paths
        from collections import deque
        
        queue = deque([(source_node, [source_node])])
        visited = {source_node}
        
        while queue:
            current_node, path = queue.popleft()
            
            if current_node == target_node:
                paths.append(path)
                continue
            
            if len(path) >= max_length:
                continue
            
            # Get neighbors
            if current_node in self.graph['customer_products']:
                neighbors = self.graph['customer_products'][current_node]
            elif current_node in self.graph['product_customers']:
                neighbors = self.graph['product_customers'][current_node]
            else:
                continue
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        logger.info(f"Found {len(paths)} paths")
        
        return paths
    
    def recommend_by_graph_structure(
        self,
        customer_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Recommend products based on graph structure (collaborative filtering).
        
        Args:
            customer_id: Customer ID
            top_k: Number of recommendations
        
        Returns:
            List of product recommendations
        """
        logger.info(f"Generating graph-based recommendations for customer {customer_id}...")
        
        if customer_id not in self.graph['customer_products']:
            logger.warning(f"Customer {customer_id} not found in graph")
            return []
        
        # Get products purchased by customer
        customer_products = self.graph['customer_products'][customer_id]
        
        # Find similar customers (customers who purchased same products)
        similar_customers = set()
        for product in customer_products:
            similar_customers.update(self.graph['product_customers'][product])
        
        similar_customers.discard(customer_id)
        
        # Get products purchased by similar customers
        product_scores = Counter()
        for similar_customer in similar_customers:
            for product in self.graph['customer_products'][similar_customer]:
                if product not in customer_products:
                    product_scores[product] += 1
        
        # Get top recommendations
        top_products = product_scores.most_common(top_k)
        
        recommendations = [
            {
                'product_id': product,
                'score': score,
                'n_similar_customers': score
            }
            for product, score in top_products
        ]
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def analyze_product_co_purchases(
        self,
        min_co_occurrence: int = 5
    ) -> pd.DataFrame:
        """
        Analyze product co-purchase patterns.
        
        Args:
            min_co_occurrence: Minimum co-occurrence threshold
        
        Returns:
            DataFrame with co-purchase analysis
        """
        logger.info("Analyzing product co-purchase patterns...")
        
        co_purchases = defaultdict(Counter)
        
        # Count co-purchases for each customer
        for customer_id, products in self.graph['customer_products'].items():
            product_list = list(products)
            for i, product1 in enumerate(product_list):
                for product2 in product_list[i+1:]:
                    co_purchases[product1][product2] += 1
                    co_purchases[product2][product1] += 1
        
        # Convert to DataFrame
        co_purchase_data = []
        for product1, related_products in co_purchases.items():
            for product2, count in related_products.items():
                if count >= min_co_occurrence:
                    co_purchase_data.append({
                        'product1': product1,
                        'product2': product2,
                        'co_occurrence_count': count
                    })
        
        co_purchase_df = pd.DataFrame(co_purchase_data)
        
        logger.info(f"Co-purchase analysis complete. {len(co_purchase_df)} pairs found")
        
        return co_purchase_df
    
    def get_graph_statistics(self) -> Dict:
        """
        Get comprehensive graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        n_customers = len(self.graph['customer_products'])
        n_products = len(self.graph['product_customers'])
        n_edges = sum(len(products) for products in self.graph['customer_products'].values())
        
        # Calculate density
        max_possible_edges = n_customers * n_products
        density = n_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Calculate clustering coefficient (simplified)
        customer_degrees = [len(products) for products in self.graph['customer_products'].values()]
        avg_degree = np.mean(customer_degrees) if customer_degrees else 0
        
        stats = {
            'n_nodes': n_customers + n_products,
            'n_customers': n_customers,
            'n_products': n_products,
            'n_edges': n_edges,
            'graph_density': float(density),
            'avg_degree': float(avg_degree),
            'is_bipartite': True
        }
        
        return stats


def run_graph_analytics_pipeline(
    orders_df: pd.DataFrame,
    customer_col: str = 'customer_id',
    product_col: str = 'product_id'
) -> Tuple[GraphAnalytics, Dict]:
    """
    Convenience function to run complete graph analytics pipeline.
    
    Args:
        orders_df: Order data
        customer_col: Customer ID column name
        product_col: Product ID column name
    
    Returns:
        Tuple of (graph_analytics, results)
    """
    analytics = GraphAnalytics()
    
    # Build graph
    graph_stats = analytics.build_customer_product_graph(orders_df, customer_col, product_col)
    
    # Calculate centrality
    customer_centrality = analytics.calculate_centrality('customer')
    product_centrality = analytics.calculate_centrality('product')
    
    # Detect communities
    customer_communities = analytics.detect_communities('customer')
    product_communities = analytics.detect_communities('product')
    
    # Analyze co-purchases
    co_purchases = analytics.analyze_product_co_purchases()
    
    # Get overall statistics
    overall_stats = analytics.get_graph_statistics()
    
    results = {
        'graph_stats': graph_stats,
        'customer_centrality': customer_centrality,
        'product_centrality': product_centrality,
        'customer_communities': customer_communities,
        'product_communities': product_communities,
        'co_purchases': co_purchases,
        'overall_stats': overall_stats
    }
    
    return analytics, results
