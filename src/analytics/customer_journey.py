"""
Customer Journey and Funnel Analytics Module
Implements customer journey mapping and funnel analysis for conversion optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from config.logging_config import get_logger

logger = get_logger(__name__)


class CustomerJourneyAnalyzer:
    """
    Customer journey and funnel analytics engine.
    
    Features:
    - Funnel stage analysis (awareness, consideration, conversion, retention)
    - Journey path mapping
    - Drop-off analysis
    - Conversion rate optimization
    - Channel attribution
    """
    
    def __init__(self):
        """Initialize customer journey analyzer"""
        self.funnel_stages = ['awareness', 'consideration', 'add_to_cart', 'checkout', 'purchase', 'retention']
        self.journey_paths = {}
    
    def define_funnel_stages(self, stages: List[str]) -> None:
        """
        Define custom funnel stages.
        
        Args:
            stages: List of funnel stage names
        """
        self.funnel_stages = stages
        logger.info(f"Funnel stages defined: {stages}")
    
    def analyze_funnel(
        self,
        events_df: pd.DataFrame,
        customer_col: str = 'customer_id',
        event_col: str = 'event_type',
        date_col: str = 'event_date'
    ) -> Dict:
        """
        Analyze funnel conversion rates.
        
        Args:
            events_df: DataFrame with customer events
            customer_col: Column name for customer ID
            event_col: Column name for event type
            date_col: Column name for event date
        
        Returns:
            Dictionary with funnel analysis results
        """
        logger.info("Analyzing funnel conversion rates...")
        
        # Count customers at each stage
        funnel_counts = {}
        stage_customers = {}
        
        for stage in self.funnel_stages:
            stage_events = events_df[events_df[event_col] == stage]
            unique_customers = stage_events[customer_col].nunique()
            funnel_counts[stage] = unique_customers
            stage_customers[stage] = set(stage_events[customer_col].unique())
        
        # Calculate conversion rates
        conversion_rates = {}
        drop_off_rates = {}
        
        for i in range(len(self.funnel_stages) - 1):
            current_stage = self.funnel_stages[i]
            next_stage = self.funnel_stages[i + 1]
            
            if funnel_counts[current_stage] > 0:
                conversion_rate = funnel_counts[next_stage] / funnel_counts[current_stage]
                conversion_rates[f"{current_stage}_to_{next_stage}"] = conversion_rate
                drop_off_rates[f"{current_stage}_to_{next_stage}"] = 1 - conversion_rate
        
        # Overall conversion rate
        if funnel_counts[self.funnel_stages[0]] > 0:
            overall_conversion = funnel_counts[self.funnel_stages[-1]] / funnel_counts[self.funnel_stages[0]]
        else:
            overall_conversion = 0
        
        results = {
            'funnel_counts': funnel_counts,
            'conversion_rates': conversion_rates,
            'drop_off_rates': drop_off_rates,
            'overall_conversion_rate': overall_conversion,
            'n_stages': len(self.funnel_stages)
        }
        
        logger.info(f"Funnel analysis complete. Overall conversion: {overall_conversion:.2%}")
        
        return results
    
    def analyze_journey_paths(
        self,
        events_df: pd.DataFrame,
        customer_col: str = 'customer_id',
        event_col: str = 'event_type',
        date_col: str = 'event_date',
        max_path_length: int = 10
    ) -> Dict:
        """
        Analyze customer journey paths.
        
        Args:
            events_df: DataFrame with customer events
            customer_col: Column name for customer ID
            event_col: Column name for event type
            date_col: Column name for event date
            max_path_length: Maximum path length to consider
        
        Returns:
            Dictionary with journey path analysis
        """
        logger.info("Analyzing customer journey paths...")
        
        # Sort events by customer and date
        events_df = events_df.sort_values([customer_col, date_col])
        
        # Extract journey paths for each customer
        journey_paths = []
        path_counts = defaultdict(int)
        
        for customer_id in events_df[customer_col].unique():
            customer_events = events_df[events_df[customer_col] == customer_id]
            path = customer_events[event_col].tolist()
            
            # Truncate to max length
            if len(path) > max_path_length:
                path = path[:max_path_length]
            
            # Convert to string for counting
            path_str = ' -> '.join(path)
            path_counts[path_str] += 1
            journey_paths.append({
                'customer_id': customer_id,
                'path': path,
                'path_length': len(path),
                'final_stage': path[-1] if path else None
            })
        
        # Find most common paths
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
        top_paths = sorted_paths[:10]
        
        # Calculate path statistics
        path_lengths = [jp['path_length'] for jp in journey_paths]
        
        results = {
            'n_unique_paths': len(path_counts),
            'top_paths': [{'path': path, 'count': count} for path, count in top_paths],
            'avg_path_length': float(np.mean(path_lengths)),
            'median_path_length': float(np.median(path_lengths)),
            'max_path_length': int(max(path_lengths)) if path_lengths else 0,
            'journey_paths': journey_paths
        }
        
        logger.info(f"Journey path analysis complete. {len(path_counts)} unique paths found")
        
        return results
    
    def identify_drop_off_points(
        self,
        events_df: pd.DataFrame,
        customer_col: str = 'customer_id',
        event_col: str = 'event_type',
        date_col: str = 'event_date'
    ) -> Dict:
        """
        Identify points where customers drop off in the funnel.
        
        Args:
            events_df: DataFrame with customer events
            customer_col: Column name for customer ID
            event_col: Column name for event type
            date_col: Column name for event date
        
        Returns:
            Dictionary with drop-off analysis
        """
        logger.info("Identifying drop-off points...")
        
        # Get customers at each stage
        stage_customers = {}
        for stage in self.funnel_stages:
            stage_customers[stage] = set(
                events_df[events_df[event_col] == stage][customer_col].unique()
            )
        
        # Identify drop-offs
        drop_offs = []
        
        for i in range(len(self.funnel_stages) - 1):
            current_stage = self.funnel_stages[i]
            next_stage = self.funnel_stages[i + 1]
            
            # Customers who reached current stage but not next stage
            dropped_off = stage_customers[current_stage] - stage_customers[next_stage]
            
            if len(stage_customers[current_stage]) > 0:
                drop_off_rate = len(dropped_off) / len(stage_customers[current_stage])
            else:
                drop_off_rate = 0
            
            drop_offs.append({
                'from_stage': current_stage,
                'to_stage': next_stage,
                'n_dropped_off': len(dropped_off),
                'drop_off_rate': drop_off_rate,
                'n_remaining': len(stage_customers[next_stage])
            })
        
        # Find largest drop-off
        largest_drop_off = max(drop_offs, key=lambda x: x['drop_off_rate']) if drop_offs else None
        
        results = {
            'drop_offs': drop_offs,
            'largest_drop_off': largest_drop_off,
            'total_drop_offs': sum(d['n_dropped_off'] for d in drop_offs)
        }
        
        logger.info(f"Drop-off analysis complete. Largest drop-off: {largest_drop_off['from_stage']} -> {largest_drop_off['to_stage']} ({largest_drop_off['drop_off_rate']:.1%})")
        
        return results
    
    def analyze_time_in_stage(
        self,
        events_df: pd.DataFrame,
        customer_col: str = 'customer_id',
        event_col: str = 'event_type',
        date_col: str = 'event_date'
    ) -> Dict:
        """
        Analyze time spent in each funnel stage.
        
        Args:
            events_df: DataFrame with customer events
            customer_col: Column name for customer ID
            event_col: Column name for event type
            date_col: Column name for event date
        
        Returns:
            Dictionary with time-in-stage analysis
        """
        logger.info("Analyzing time spent in each stage...")
        
        events_df[date_col] = pd.to_datetime(events_df[date_col])
        
        # Calculate time between stages for each customer
        stage_times = defaultdict(list)
        
        for customer_id in events_df[customer_col].unique():
            customer_events = events_df[events_df[customer_col] == customer_id].sort_values(date_col)
            
            for i in range(len(customer_events) - 1):
                current_event = customer_events.iloc[i]
                next_event = customer_events.iloc[i + 1]
                
                current_stage = current_event[event_col]
                next_stage = next_event[event_col]
                
                # Only track transitions between funnel stages
                if current_stage in self.funnel_stages and next_stage in self.funnel_stages:
                    time_diff = (next_event[date_col] - current_event[date_col]).total_seconds() / 3600  # hours
                    stage_times[f"{current_stage}_to_{next_stage}"].append(time_diff)
        
        # Calculate statistics
        stage_stats = {}
        for transition, times in stage_times.items():
            if times:
                stage_stats[transition] = {
                    'mean_hours': float(np.mean(times)),
                    'median_hours': float(np.median(times)),
                    'std_hours': float(np.std(times)),
                    'min_hours': float(np.min(times)),
                    'max_hours': float(np.max(times)),
                    'n_transitions': len(times)
                }
        
        results = {
            'stage_times': stage_stats,
            'n_transitions_analyzed': sum(len(times) for times in stage_times.values())
        }
        
        logger.info(f"Time-in-stage analysis complete. Analyzed {results['n_transitions_analyzed']} transitions")
        
        return results
    
    def calculate_funnel_value(
        self,
        events_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        customer_col: str = 'customer_id',
        event_col: str = 'event_type',
        date_col: str = 'event_date',
        value_col: str = 'order_total'
    ) -> Dict:
        """
        Calculate value at each funnel stage.
        
        Args:
            events_df: DataFrame with customer events
            orders_df: DataFrame with order data
            customer_col: Column name for customer ID
            event_col: Column name for event type
            date_col: Column name for event date
            value_col: Column name for order value
        
        Returns:
            Dictionary with funnel value analysis
        """
        logger.info("Calculating funnel value...")
        
        # Get customers at each stage
        stage_customers = {}
        for stage in self.funnel_stages:
            stage_customers[stage] = list(
                events_df[events_df[event_col] == stage][customer_col].unique()
            )
        
        # Calculate total value for customers at each stage
        stage_values = {}
        
        for stage in self.funnel_stages:
            customers = stage_customers[stage]
            if customers:
                customer_orders = orders_df[orders_df[customer_col].isin(customers)]
                total_value = customer_orders[value_col].sum()
                avg_value = customer_orders[value_col].mean()
                
                stage_values[stage] = {
                    'total_value': float(total_value),
                    'avg_value_per_customer': float(avg_value),
                    'n_customers': len(customers)
                }
            else:
                stage_values[stage] = {
                    'total_value': 0.0,
                    'avg_value_per_customer': 0.0,
                    'n_customers': 0
                }
        
        results = {
            'stage_values': stage_values,
            'total_value': stage_values[self.funnel_stages[-1]]['total_value']
        }
        
        logger.info(f"Funnel value analysis complete. Total value: ${results['total_value']:,.2f}")
        
        return results
    
    def segment_journeys(
        self,
        events_df: pd.DataFrame,
        customer_features_df: pd.DataFrame,
        customer_col: str = 'customer_id',
        event_col: str = 'event_type',
        segment_col: str = 'segment'
    ) -> Dict:
        """
        Analyze journeys by customer segment.
        
        Args:
            events_df: DataFrame with customer events
            customer_features_df: DataFrame with customer features
            customer_col: Column name for customer ID
            event_col: Column name for event type
            segment_col: Column name for segment
        
        Returns:
            Dictionary with segmented journey analysis
        """
        logger.info("Analyzing journeys by segment...")
        
        # Merge events with customer features
        events_with_segment = events_df.merge(
            customer_features_df[[customer_col, segment_col]],
            on=customer_col,
            how='left'
        )
        
        # Analyze funnel for each segment
        segment_funnels = {}
        
        for segment in events_with_segment[segment_col].unique():
            segment_events = events_with_segment[events_with_segment[segment_col] == segment]
            funnel_analysis = self.analyze_funnel(segment_events, customer_col, event_col)
            segment_funnels[segment] = funnel_analysis
        
        # Compare segments
        segment_comparison = []
        for segment, analysis in segment_funnels.items():
            segment_comparison.append({
                'segment': segment,
                'overall_conversion_rate': analysis['overall_conversion_rate'],
                'n_customers': analysis['funnel_counts'][self.funnel_stages[0]]
            })
        
        segment_comparison_df = pd.DataFrame(segment_comparison)
        segment_comparison_df = segment_comparison_df.sort_values('overall_conversion_rate', ascending=False)
        
        results = {
            'segment_funnels': segment_funnels,
            'segment_comparison': segment_comparison_df.to_dict('records')
        }
        
        logger.info(f"Segmented journey analysis complete. {len(segment_funnels)} segments analyzed")
        
        return results


def run_customer_journey_pipeline(
    events_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    customer_features_df: pd.DataFrame = None,
    funnel_stages: List[str] = None
) -> Tuple[CustomerJourneyAnalyzer, Dict]:
    """
    Convenience function to run complete customer journey pipeline.
    
    Args:
        events_df: Customer events data
        orders_df: Order data
        customer_features_df: Customer features (optional)
        funnel_stages: Custom funnel stages (optional)
    
    Returns:
        Tuple of (analyzer, analysis results)
    """
    analyzer = CustomerJourneyAnalyzer()
    
    if funnel_stages:
        analyzer.define_funnel_stages(funnel_stages)
    
    # Analyze funnel
    funnel_analysis = analyzer.analyze_funnel(events_df)
    
    # Analyze journey paths
    journey_paths = analyzer.analyze_journey_paths(events_df)
    
    # Identify drop-offs
    drop_offs = analyzer.identify_drop_off_points(events_df)
    
    # Analyze time in stage
    time_in_stage = analyzer.analyze_time_in_stage(events_df)
    
    # Calculate funnel value
    funnel_value = analyzer.calculate_funnel_value(events_df, orders_df)
    
    # Segment journeys if features available
    segmented_journeys = None
    if customer_features_df is not None:
        segmented_journeys = analyzer.segment_journeys(events_df, customer_features_df)
    
    results = {
        'funnel_analysis': funnel_analysis,
        'journey_paths': journey_paths,
        'drop_offs': drop_offs,
        'time_in_stage': time_in_stage,
        'funnel_value': funnel_value,
        'segmented_journeys': segmented_journeys
    }
    
    return analyzer, results
