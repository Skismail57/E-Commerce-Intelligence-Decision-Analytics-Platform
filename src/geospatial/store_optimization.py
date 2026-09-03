"""
Geospatial Intelligence and Store Optimization Module
Implements geospatial analysis for store location optimization and territory management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from config.logging_config import get_logger

logger = get_logger(__name__)


class GeospatialOptimizer:
    """
    Geospatial intelligence engine for store optimization.
    
    Features:
    - Store location optimization
    - Territory analysis and clustering
    - Trade area analysis
    - Competitor proximity analysis
    - Demand density mapping
    """
    
    def __init__(self):
        """Initialize geospatial optimizer"""
        self.store_clusters = None
        self.scaler = None
    
    def analyze_trade_area(
        self,
        stores_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        store_id_col: str = 'store_id',
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        radius_km: float = 10.0
    ) -> Dict:
        """
        Analyze trade areas for stores.
        
        Args:
            stores_df: DataFrame with store locations
            customers_df: DataFrame with customer locations
            store_id_col: Store ID column name
            lat_col: Latitude column name
            lon_col: Longitude column name
            radius_km: Trade area radius in km
        
        Returns:
            Dictionary with trade area analysis
        """
        logger.info("Analyzing trade areas...")
        
        trade_areas = {}
        
        for _, store in stores_df.iterrows():
            store_id = store[store_id_col]
            store_lat = store[lat_col]
            store_lon = store[lon_col]
            
            # Calculate distance to each customer
            customer_lats = customers_df[lat_col].values
            customer_lons = customers_df[lon_col].values
            
            # Haversine distance approximation
            distances = self._haversine_distance(
                store_lat, store_lon, customer_lats, customer_lons
            )
            
            # Customers within trade area
            in_trade_area = distances <= radius_km
            trade_area_customers = customers_df[in_trade_area]
            
            trade_areas[store_id] = {
                'n_customers': len(trade_area_customers),
                'trade_area_radius_km': radius_km,
                'avg_distance_km': float(distances[in_trade_area].mean()) if in_trade_area.sum() > 0 else 0,
                'customer_ids': trade_area_customers['customer_id'].tolist() if 'customer_id' in trade_area_customers.columns else []
            }
        
        logger.info(f"Trade area analysis complete for {len(trade_areas)} stores")
        
        return trade_areas
    
    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: np.ndarray,
        lon2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Haversine distance between points.
        
        Args:
            lat1: Latitude of point 1
            lon1: Longitude of point 1
            lat2: Array of latitudes
            lon2: Array of longitudes
        
        Returns:
            Array of distances in km
        """
        R = 6371  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        distances = R * c
        
        return distances
    
    def optimize_store_locations(
        self,
        demand_points: pd.DataFrame,
        n_stores: int = 5,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        demand_col: str = 'demand'
    ) -> Dict:
        """
        Optimize store locations using K-means clustering.
        
        Args:
            demand_points: DataFrame with demand points
            n_stores: Number of stores to locate
            lat_col: Latitude column name
            lon_col: Longitude column name
            demand_col: Demand column name
        
        Returns:
            Dictionary with optimal store locations
        """
        logger.info(f"Optimizing {n_stores} store locations...")
        
        # Prepare features
        features = demand_points[[lat_col, lon_col]].values
        weights = demand_points[demand_col].values if demand_col in demand_points.columns else np.ones(len(demand_points))
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Weighted K-means
        weighted_features = features_scaled * weights.reshape(-1, 1)
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_stores, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(weighted_features)
        
        # Get cluster centers (inverse transform)
        centers_scaled = kmeans.cluster_centers_
        centers = self.scaler.inverse_transform(centers_scaled)
        
        # Calculate store coverage
        store_locations = []
        for i in range(n_stores):
            store_lat = centers[i][0]
            store_lon = centers[i][1]
            
            # Calculate distance to all demand points
            distances = self._haversine_distance(store_lat, store_lon, features[:, 0], features[:, 1])
            
            # Assign each demand point to nearest store
            assigned_to_store = clusters == i
            
            store_locations.append({
                'store_id': i,
                'latitude': float(store_lat),
                'longitude': float(store_lon),
                'n_assigned_customers': int(assigned_to_store.sum()),
                'total_demand': float(weights[assigned_to_store].sum()),
                'avg_distance_km': float(distances[assigned_to_store].mean()) if assigned_to_store.sum() > 0 else 0
            })
        
        results = {
            'n_stores': n_stores,
            'store_locations': store_locations,
            'total_demand_covered': float(weights.sum()),
            'avg_coverage_distance': float(np.mean([s['avg_distance_km'] for s in store_locations]))
        }
        
        logger.info(f"Store location optimization complete")
        
        return results
    
    def analyze_competitor_proximity(
        self,
        stores_df: pd.DataFrame,
        competitors_df: pd.DataFrame,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        proximity_threshold_km: float = 5.0
    ) -> pd.DataFrame:
        """
        Analyze proximity to competitors.
        
        Args:
            stores_df: DataFrame with store locations
            competitors_df: DataFrame with competitor locations
            lat_col: Latitude column name
            lon_col: Longitude column name
            proximity_threshold_km: Proximity threshold in km
        
        Returns:
            DataFrame with competitor proximity analysis
        """
        logger.info("Analyzing competitor proximity...")
        
        results = []
        
        for _, store in stores_df.iterrows():
            store_lat = store[lat_col]
            store_lon = store[lon_col]
            
            # Calculate distance to each competitor
            comp_lats = competitors_df[lat_col].values
            comp_lons = competitors_df[lon_col].values
            
            distances = self._haversine_distance(store_lat, store_lon, comp_lats, comp_lons)
            
            # Find nearest competitor
            nearest_dist = distances.min()
            nearest_idx = distances.argmin()
            
            # Count competitors within threshold
            nearby_competitors = (distances <= proximity_threshold_km).sum()
            
            results.append({
                'store_id': store.get('store_id', ''),
                'nearest_competitor_distance_km': float(nearest_dist),
                'n_nearby_competitors': int(nearby_competitors),
                'competitor_saturation': 'high' if nearby_competitors >= 3 else 'medium' if nearby_competitors >= 1 else 'low'
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Competitor proximity analysis complete for {len(results_df)} stores")
        
        return results_df
    
    def cluster_territories(
        self,
        customers_df: pd.DataFrame,
        n_territories: int = 10,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude'
    ) -> Dict:
        """
        Cluster customers into territories.
        
        Args:
            customers_df: DataFrame with customer locations
            n_territories: Number of territories
            lat_col: Latitude column name
            lon_col: Longitude column name
        
        Returns:
            Dictionary with territory clusters
        """
        logger.info(f"Clustering customers into {n_territories} territories...")
        
        # Prepare features
        features = customers_df[[lat_col, lon_col]].values
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_territories, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Get cluster centers
        centers_scaled = kmeans.cluster_centers_
        centers = self.scaler.inverse_transform(centers_scaled)
        
        # Analyze each territory
        territories = []
        for i in range(n_territories):
            territory_customers = customers_df[clusters == i]
            
            # Calculate territory bounds
            min_lat = territory_customers[lat_col].min()
            max_lat = territory_customers[lat_col].max()
            min_lon = territory_customers[lon_col].min()
            max_lon = territory_customers[lon_col].max()
            
            territories.append({
                'territory_id': i,
                'center_lat': float(centers[i][0]),
                'center_lon': float(centers[i][1]),
                'n_customers': len(territory_customers),
                'min_lat': float(min_lat),
                'max_lat': float(max_lat),
                'min_lon': float(min_lon),
                'max_lon': float(max_lon)
            })
        
        results = {
            'n_territories': n_territories,
            'territories': territories,
            'customer_assignments': clusters.tolist()
        }
        
        logger.info(f"Territory clustering complete")
        
        return results
    
    def calculate_demand_density(
        self,
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        grid_size_km: float = 5.0
    ) -> pd.DataFrame:
        """
        Calculate demand density on a grid.
        
        Args:
            customers_df: DataFrame with customer locations
            orders_df: DataFrame with order data
            lat_col: Latitude column name
            lon_col: Longitude column name
            grid_size_km: Grid cell size in km
        
        Returns:
            DataFrame with demand density grid
        """
        logger.info("Calculating demand density...")
        
        # Calculate customer spend
        customer_spend = orders_df.groupby('customer_id')['order_total'].sum()
        customers_df = customers_df.merge(customer_spend, on='customer_id', how='left').fillna(0)
        
        # Create grid
        min_lat = customers_df[lat_col].min()
        max_lat = customers_df[lat_col].max()
        min_lon = customers_df[lon_col].min()
        max_lon = customers_df[lon_col].max()
        
        # Convert km to degrees (approximate)
        lat_deg_per_km = 1 / 111.0
        lon_deg_per_km = 1 / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))
        
        lat_step = grid_size_km * lat_deg_per_km
        lon_step = grid_size_km * lon_deg_per_km
        
        # Create grid cells
        grid_cells = []
        lat = min_lat
        while lat < max_lat:
            lon = min_lon
            while lon < max_lon:
                # Find customers in this cell
                in_cell = (
                    (customers_df[lat_col] >= lat) &
                    (customers_df[lat_col] < lat + lat_step) &
                    (customers_df[lon_col] >= lon) &
                    (customers_df[lon_col] < lon + lon_step)
                )
                
                cell_customers = customers_df[in_cell]
                total_demand = cell_customers['order_total'].sum()
                
                grid_cells.append({
                    'grid_lat': lat,
                    'grid_lon': lon,
                    'n_customers': len(cell_customers),
                    'total_demand': float(total_demand),
                    'demand_density': float(total_demand / len(cell_customers)) if len(cell_customers) > 0 else 0
                })
                
                lon += lon_step
            lat += lat_step
        
        grid_df = pd.DataFrame(grid_cells)
        
        logger.info(f"Demand density calculated for {len(grid_df)} grid cells")
        
        return grid_df


def run_geospatial_pipeline(
    stores_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    competitors_df: pd.DataFrame = None,
    n_stores: int = 5
) -> Tuple[GeospatialOptimizer, Dict]:
    """
    Convenience function to run complete geospatial pipeline.
    
    Args:
        stores_df: Store data
        customers_df: Customer data
        orders_df: Order data
        competitors_df: Competitor data (optional)
        n_stores: Number of stores for optimization
    
    Returns:
        Tuple of (optimizer, results)
    """
    optimizer = GeospatialOptimizer()
    
    # Analyze trade areas
    trade_areas = optimizer.analyze_trade_area(stores_df, customers_df)
    
    # Calculate demand density
    demand_density = optimizer.calculate_demand_density(customers_df, orders_df)
    
    # Analyze competitor proximity if data available
    competitor_proximity = None
    if competitors_df is not None:
        competitor_proximity = optimizer.analyze_competitor_proximity(stores_df, competitors_df)
    
    # Cluster territories
    territories = optimizer.cluster_territories(customers_df)
    
    results = {
        'trade_areas': trade_areas,
        'demand_density': demand_density,
        'competitor_proximity': competitor_proximity,
        'territories': territories
    }
    
    return optimizer, results
