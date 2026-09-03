"""
Executive Decision Center 2.0 Module
Implements upgraded executive decision center with advanced analytics and insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from config.logging_config import get_logger

logger = get_logger(__name__)


class ExecutiveDecisionCenter:
    """
    Executive Decision Center 2.0 for comprehensive business intelligence.
    
    Features:
    - KPI dashboard aggregation
    - Executive summary generation
    - Trend analysis and forecasting
    - Risk assessment
    - Opportunity identification
    - Actionable recommendations
    """
    
    def __init__(self):
        """Initialize Executive Decision Center"""
        self.kpis = {}
        self.alerts = []
        self.recommendations = []
        logger.info("Executive Decision Center 2.0 initialized")
    
    def register_kpi(
        self,
        kpi_name: str,
        kpi_value: float,
        target: float = None,
        unit: str = None,
        category: str = 'general'
    ) -> Dict:
        """
        Register a KPI.
        
        Args:
            kpi_name: Name of the KPI
            kpi_value: Current KPI value
            target: Target value
            unit: Unit of measurement
            category: KPI category
        
        Returns:
            Dictionary with KPI information
        """
        logger.info(f"Registering KPI: {kpi_name}")
        
        kpi = {
            'name': kpi_name,
            'value': kpi_value,
            'target': target,
            'unit': unit,
            'category': category,
            'variance': (kpi_value - target) / target if target and target != 0 else None,
            'updated_at': datetime.now().isoformat()
        }
        
        self.kpis[kpi_name] = kpi
        
        # Generate alert if variance is significant
        if kpi['variance'] and abs(kpi['variance']) > 0.1:
            self.alerts.append({
                'kpi': kpi_name,
                'type': 'variance_alert',
                'message': f"{kpi_name} is {kpi['variance']*100:.1f}% {'above' if kpi['variance'] > 0 else 'below'} target",
                'severity': 'high' if abs(kpi['variance']) > 0.2 else 'medium',
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"KPI {kpi_name} registered with value {kpi_value}")
        
        return kpi
    
    def calculate_kpi_trend(
        self,
        kpi_name: str,
        historical_values: List[float],
        periods: int = 7
    ) -> Dict:
        """
        Calculate KPI trend over time.
        
        Args:
            kpi_name: Name of the KPI
            historical_values: List of historical values
            periods: Number of periods to analyze
        
        Returns:
            Dictionary with trend analysis
        """
        logger.info(f"Calculating trend for {kpi_name}")
        
        if len(historical_values) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        values = historical_values[-periods:]
        
        # Calculate trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Determine trend direction
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate growth rate
        if values[0] != 0:
            growth_rate = (values[-1] - values[0]) / abs(values[0])
        else:
            growth_rate = 0
        
        # Calculate volatility
        volatility = np.std(values)
        
        # Forecast next period
        forecast = values[-1] + slope
        
        trend_info = {
            'kpi_name': kpi_name,
            'trend': trend,
            'slope': float(slope),
            'growth_rate': float(growth_rate),
            'volatility': float(volatility),
            'current_value': float(values[-1]),
            'forecast': float(forecast),
            'n_periods': len(values)
        }
        
        logger.info(f"Trend calculated for {kpi_name}: {trend}")
        
        return trend_info
    
    def assess_risk(
        self,
        risk_factors: Dict[str, float],
        thresholds: Dict[str, float] = None
    ) -> Dict:
        """
        Assess overall business risk.
        
        Args:
            risk_factors: Dictionary of risk factor names and values
            thresholds: Dictionary of risk thresholds
        
        Returns:
            Dictionary with risk assessment
        """
        logger.info("Assessing business risk...")
        
        if thresholds is None:
            thresholds = {k: 0.7 for k in risk_factors.keys()}
        
        risk_scores = {}
        total_risk = 0
        
        for factor, value in risk_factors.items():
            threshold = thresholds.get(factor, 0.7)
            risk_score = min(1.0, value / threshold) if threshold > 0 else 0
            risk_scores[factor] = {
                'value': value,
                'threshold': threshold,
                'risk_score': float(risk_score),
                'severity': 'high' if risk_score > 0.8 else 'medium' if risk_score > 0.5 else 'low'
            }
            total_risk += risk_score
        
        # Calculate overall risk
        overall_risk = total_risk / len(risk_factors) if risk_factors else 0
        
        risk_assessment = {
            'overall_risk': float(overall_risk),
            'risk_level': 'critical' if overall_risk > 0.8 else 'high' if overall_risk > 0.6 else 'medium' if overall_risk > 0.4 else 'low',
            'risk_factors': risk_scores,
            'n_factors': len(risk_factors)
        }
        
        logger.info(f"Risk assessment complete. Overall risk: {overall_risk:.3f}")
        
        return risk_assessment
    
    def identify_opportunities(
        self,
        metrics: Dict[str, float],
        targets: Dict[str, float]
    ) -> List[Dict]:
        """
        Identify business opportunities based on metrics vs targets.
        
        Args:
            metrics: Current metrics
            targets: Target values
        
        Returns:
            List of opportunity recommendations
        """
        logger.info("Identifying business opportunities...")
        
        opportunities = []
        
        for metric, current_value in metrics.items():
            if metric in targets:
                target = targets[metric]
                gap = target - current_value
                
                if gap > 0:
                    priority = 'high' if gap / target > 0.2 else 'medium' if gap / target > 0.1 else 'low'
                    
                    opportunities.append({
                        'metric': metric,
                        'current_value': current_value,
                        'target': target,
                        'gap': float(gap),
                        'gap_percentage': float(gap / target * 100) if target != 0 else 0,
                        'priority': priority,
                        'potential_impact': 'high' if gap / target > 0.2 else 'medium'
                    })
        
        # Sort by gap percentage
        opportunities.sort(key=lambda x: x['gap_percentage'], reverse=True)
        
        self.recommendations.extend(opportunities)
        
        logger.info(f"Identified {len(opportunities)} opportunities")
        
        return opportunities
    
    def generate_executive_summary(
        self,
        kpis: Dict[str, float],
        trends: Dict[str, Dict] = None,
        risk_assessment: Dict = None
    ) -> Dict:
        """
        Generate executive summary.
        
        Args:
            kpis: Dictionary of KPIs
            trends: Dictionary of trend analyses
            risk_assessment: Risk assessment results
        
        Returns:
            Dictionary with executive summary
        """
        logger.info("Generating executive summary...")
        
        # Calculate overall performance
        kpi_values = list(kpis.values())
        avg_performance = np.mean(kpi_values) if kpi_values else 0
        
        # Count positive and negative trends
        positive_trends = 0
        negative_trends = 0
        
        if trends:
            for trend_info in trends.values():
                if trend_info.get('trend') == 'increasing':
                    positive_trends += 1
                elif trend_info.get('trend') == 'decreasing':
                    negative_trends += 1
        
        # Determine overall health
        health_score = avg_performance
        if risk_assessment:
            health_score -= risk_assessment.get('overall_risk', 0) * 0.3
        
        health_status = 'excellent' if health_score > 0.8 else 'good' if health_score > 0.6 else 'fair' if health_score > 0.4 else 'poor'
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'overall_performance': float(avg_performance),
            'health_status': health_status,
            'health_score': float(health_score),
            'n_kpis': len(kpis),
            'positive_trends': positive_trends,
            'negative_trends': negative_trends,
            'n_alerts': len(self.alerts),
            'n_recommendations': len(self.recommendations),
            'top_alerts': self.alerts[-5:] if self.alerts else [],
            'top_recommendations': self.recommendations[:5] if self.recommendations else [],
            'risk_level': risk_assessment.get('risk_level', 'unknown') if risk_assessment else 'unknown'
        }
        
        logger.info("Executive summary generated")
        
        return summary
    
    def create_action_plan(
        self,
        opportunities: List[Dict],
        constraints: Dict = None
    ) -> List[Dict]:
        """
        Create actionable plan from opportunities.
        
        Args:
            opportunities: List of opportunity recommendations
            constraints: Resource constraints
        
        Returns:
            List of action items
        """
        logger.info("Creating action plan...")
        
        action_plan = []
        
        for i, opportunity in enumerate(opportunities):
            action_item = {
                'action_id': f"ACTION_{i+1}",
                'metric': opportunity['metric'],
                'action': f"Improve {opportunity['metric']} to reach target",
                'current_value': opportunity['current_value'],
                'target_value': opportunity['target'],
                'gap': opportunity['gap'],
                'priority': opportunity['priority'],
                'estimated_effort': 'high' if opportunity['gap_percentage'] > 20 else 'medium' if opportunity['gap_percentage'] > 10 else 'low',
                'timeline': f"{opportunity['gap_percentage'] // 10 + 1} weeks" if opportunity['gap_percentage'] > 0 else '1 week',
                'owner': 'TBD',
                'status': 'pending'
            }
            action_plan.append(action_item)
        
        logger.info(f"Action plan created with {len(action_plan)} items")
        
        return action_plan
    
    def get_dashboard_data(self) -> Dict:
        """
        Get complete dashboard data.
        
        Returns:
            Dictionary with all dashboard data
        """
        dashboard_data = {
            'kpis': self.kpis,
            'alerts': self.alerts,
            'recommendations': self.recommendations,
            'n_kpis': len(self.kpis),
            'n_alerts': len(self.alerts),
            'n_recommendations': len(self.recommendations),
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard_data


def run_executive_center_pipeline(
    kpis: Dict[str, float],
    targets: Dict[str, float],
    historical_data: Dict[str, List[float]] = None
) -> Tuple[ExecutiveDecisionCenter, Dict]:
    """
    Convenience function to run Executive Decision Center pipeline.
    
    Args:
        kpis: Dictionary of current KPI values
        targets: Dictionary of target values
        historical_data: Historical KPI data for trend analysis
    
    Returns:
        Tuple of (center, results)
    """
    center = ExecutiveDecisionCenter()
    
    # Register KPIs
    for kpi_name, value in kpis.items():
        center.register_kpi(kpi_name, value, targets.get(kpi_name))
    
    # Calculate trends
    trends = {}
    if historical_data:
        for kpi_name, values in historical_data.items():
            trends[kpi_name] = center.calculate_kpi_trend(kpi_name, values)
    
    # Identify opportunities
    opportunities = center.identify_opportunities(kpis, targets)
    
    # Create action plan
    action_plan = center.create_action_plan(opportunities)
    
    # Generate executive summary
    summary = center.generate_executive_summary(kpis, trends)
    
    # Get dashboard data
    dashboard_data = center.get_dashboard_data()
    
    results = {
        'trends': trends,
        'opportunities': opportunities,
        'action_plan': action_plan,
        'executive_summary': summary,
        'dashboard_data': dashboard_data
    }
    
    return center, results
