"""
NLP Review Intelligence Module
Implements aspect-based sentiment analysis for customer reviews using transformers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
from config.logging_config import get_logger

logger = get_logger(__name__)


class ReviewIntelligenceEngine:
    """
    NLP-based review intelligence engine.
    
    Features:
    - Aspect-based sentiment analysis
    - Overall sentiment classification
    - Key aspect extraction (quality, price, shipping, service)
    - Review summarization
    - Trend analysis over time
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize review intelligence engine.
        
        Args:
            model_name: Hugging Face model name for sentiment analysis
        """
        logger.info(f"Loading sentiment model: {model_name}")
        
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Define aspect keywords for extraction
        self.aspect_keywords = {
            'quality': ['quality', 'durable', 'sturdy', 'well-made', 'materials', 'build', 'construction'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'worth'],
            'shipping': ['shipping', 'delivery', 'arrived', 'packaging', 'fast', 'slow', 'ship'],
            'service': ['service', 'support', 'customer service', 'helpful', 'responsive', 'staff'],
            'fit': ['fit', 'size', 'small', 'large', 'tight', 'loose', 'true to size'],
            'appearance': ['look', 'appearance', 'design', 'style', 'color', 'aesthetic']
        }
        
        logger.info("Review intelligence engine initialized")
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a review text.
        
        Args:
            text: Review text
        
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or pd.isna(text):
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        # Truncate text if too long
        if len(text) > 512:
            text = text[:512]
        
        result = self.sentiment_pipeline(text)[0]
        
        return {
            'label': result['label'],
            'score': float(result['score']),
            'positive_probability': float(result['score']) if result['label'] == 'POSITIVE' else float(1 - result['score'])
        }
    
    def extract_aspects(self, text: str) -> Dict[str, float]:
        """
        Extract aspects from review text based on keyword matching.
        
        Args:
            text: Review text
        
        Returns:
            Dictionary mapping aspects to relevance scores
        """
        if not text or pd.isna(text):
            return {}
        
        text_lower = text.lower()
        aspects = {}
        
        for aspect, keywords in self.aspect_keywords.items():
            aspect_score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    aspect_score += 1
            
            if aspect_score > 0:
                aspects[aspect] = aspect_score
        
        return aspects
    
    def analyze_review(
        self,
        review_text: str,
        review_id: int = None
    ) -> Dict:
        """
        Perform comprehensive review analysis.
        
        Args:
            review_text: Review text
            review_id: Review identifier
        
        Returns:
            Dictionary with comprehensive review analysis
        """
        sentiment = self.analyze_sentiment(review_text)
        aspects = self.extract_aspects(review_text)
        
        # Calculate aspect sentiment (simplified - uses overall sentiment)
        aspect_sentiments = {}
        for aspect, score in aspects.items():
            aspect_sentiments[aspect] = {
                'relevance': score,
                'sentiment': sentiment['label'],
                'sentiment_score': sentiment['score']
            }
        
        result = {
            'review_id': review_id,
            'overall_sentiment': sentiment['label'],
            'sentiment_score': sentiment['score'],
            'aspects': aspect_sentiments,
            'n_aspects': len(aspects),
            'dominant_aspect': max(aspects.keys()) if aspects else None
        }
        
        return result
    
    def batch_analyze_reviews(
        self,
        reviews_df: pd.DataFrame,
        text_col: str = 'review_text',
        id_col: str = 'review_id'
    ) -> pd.DataFrame:
        """
        Analyze multiple reviews in batch.
        
        Args:
            reviews_df: DataFrame with reviews
            text_col: Column name for review text
            id_col: Column name for review ID
        
        Returns:
            DataFrame with analysis results
        """
        logger.info(f"Analyzing {len(reviews_df)} reviews...")
        
        results = []
        
        for _, row in reviews_df.iterrows():
            review_text = row[text_col]
            review_id = row[id_col] if id_col in row else None
            
            analysis = self.analyze_review(review_text, review_id)
            results.append(analysis)
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Review analysis complete. Avg sentiment score: {results_df['sentiment_score'].mean():.3f}")
        
        return results_df
    
    def calculate_aspect_aggregates(
        self,
        analysis_results_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate aggregate statistics for aspects.
        
        Args:
            analysis_results_df: DataFrame with review analysis results
        
        Returns:
            Dictionary with aspect aggregates
        """
        logger.info("Calculating aspect aggregates...")
        
        # Count aspect mentions
        aspect_counts = {}
        aspect_sentiments = {}
        
        for _, row in analysis_results_df.iterrows():
            aspects = row['aspects']
            sentiment = row['overall_sentiment']
            
            for aspect in aspects.keys():
                if aspect not in aspect_counts:
                    aspect_counts[aspect] = 0
                    aspect_sentiments[aspect] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
                
                aspect_counts[aspect] += 1
                aspect_sentiments[aspect][sentiment] += 1
        
        # Calculate sentiment distribution for each aspect
        aspect_stats = {}
        for aspect, count in aspect_counts.items():
            total = aspect_sentiments[aspect]['POSITIVE'] + aspect_sentiments[aspect]['NEGATIVE'] + aspect_sentiments[aspect]['NEUTRAL']
            
            aspect_stats[aspect] = {
                'mention_count': count,
                'positive_pct': aspect_sentiments[aspect]['POSITIVE'] / total if total > 0 else 0,
                'negative_pct': aspect_sentiments[aspect]['NEGATIVE'] / total if total > 0 else 0,
                'neutral_pct': aspect_sentiments[aspect]['NEUTRAL'] / total if total > 0 else 0,
                'sentiment_score': aspect_sentiments[aspect]['POSITIVE'] / total if total > 0 else 0
            }
        
        # Sort by mention count
        aspect_stats = dict(sorted(aspect_stats.items(), key=lambda x: x[1]['mention_count'], reverse=True))
        
        return aspect_stats
    
    def analyze_sentiment_trends(
        self,
        analysis_results_df: pd.DataFrame,
        date_col: str = 'review_date'
    ) -> pd.DataFrame:
        """
        Analyze sentiment trends over time.
        
        Args:
            analysis_results_df: DataFrame with review analysis results
            date_col: Column name for review date
        
        Returns:
            DataFrame with sentiment trends
        """
        if date_col not in analysis_results_df.columns:
            logger.warning(f"Date column {date_col} not found")
            return pd.DataFrame()
        
        # Convert sentiment to numeric
        sentiment_map = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
        analysis_results_df['sentiment_numeric'] = analysis_results_df['overall_sentiment'].map(sentiment_map)
        
        # Group by date
        trends = analysis_results_df.groupby(date_col).agg({
            'sentiment_numeric': ['mean', 'std', 'count'],
            'sentiment_score': 'mean'
        }).reset_index()
        
        trends.columns = [date_col, 'mean_sentiment', 'std_sentiment', 'n_reviews', 'mean_sentiment_score']
        
        return trends
    
    def generate_review_summary(
        self,
        analysis_results_df: pd.DataFrame,
        product_id: int = None
    ) -> Dict:
        """
        Generate summary statistics for reviews.
        
        Args:
            analysis_results_df: DataFrame with review analysis results
            product_id: Product ID (optional)
        
        Returns:
            Dictionary with review summary
        """
        logger.info("Generating review summary...")
        
        summary = {
            'product_id': product_id,
            'total_reviews': len(analysis_results_df),
            'avg_sentiment_score': float(analysis_results_df['sentiment_score'].mean()),
            'sentiment_distribution': {
                'positive': int((analysis_results_df['overall_sentiment'] == 'POSITIVE').sum()),
                'negative': int((analysis_results_df['overall_sentiment'] == 'NEGATIVE').sum()),
                'neutral': int((analysis_results_df['overall_sentiment'] == 'NEUTRAL').sum())
            },
            'aspect_aggregates': self.calculate_aspect_aggregates(analysis_results_df)
        }
        
        # Calculate positive percentage
        total = summary['sentiment_distribution']['positive'] + summary['sentiment_distribution']['negative'] + summary['sentiment_distribution']['neutral']
        summary['positive_percentage'] = (summary['sentiment_distribution']['positive'] / total * 100) if total > 0 else 0
        
        logger.info(f"Review summary generated. Positive: {summary['positive_percentage']:.1f}%")
        
        return summary
    
    def identify_key_issues(
        self,
        analysis_results_df: pd.DataFrame,
        min_mentions: int = 5
    ) -> List[Dict]:
        """
        Identify key issues from negative reviews.
        
        Args:
            analysis_results_df: DataFrame with review analysis results
            min_mentions: Minimum mentions to consider an issue
        
        Returns:
            List of identified issues
        """
        logger.info("Identifying key issues...")
        
        # Filter negative reviews
        negative_reviews = analysis_results_df[analysis_results_df['overall_sentiment'] == 'NEGATIVE']
        
        # Count aspect mentions in negative reviews
        aspect_counts = Counter()
        
        for _, row in negative_reviews.iterrows():
            aspects = row['aspects']
            for aspect in aspects.keys():
                aspect_counts[aspect] += 1
        
        # Filter by minimum mentions
        key_issues = [
            {
                'aspect': aspect,
                'mention_count': count,
                'severity': 'high' if count > min_mentions * 2 else 'medium'
            }
            for aspect, count in aspect_counts.items()
            if count >= min_mentions
        ]
        
        # Sort by mention count
        key_issues.sort(key=lambda x: x['mention_count'], reverse=True)
        
        logger.info(f"Identified {len(key_issues)} key issues")
        
        return key_issues
    
    def compare_products(
        self,
        analysis_results_by_product: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compare review sentiment across multiple products.
        
        Args:
            analysis_results_by_product: Dictionary mapping product IDs to analysis results
        
        Returns:
            DataFrame with product comparison
        """
        logger.info(f"Comparing {len(analysis_results_by_product)} products...")
        
        comparisons = []
        
        for product_id, results_df in analysis_results_by_product.items():
            summary = self.generate_review_summary(results_df, product_id)
            comparisons.append({
                'product_id': product_id,
                'total_reviews': summary['total_reviews'],
                'avg_sentiment_score': summary['avg_sentiment_score'],
                'positive_percentage': summary['positive_percentage'],
                'n_aspects_mentioned': len(summary['aspect_aggregates'])
            })
        
        comparison_df = pd.DataFrame(comparisons)
        comparison_df = comparison_df.sort_values('avg_sentiment_score', ascending=False)
        
        return comparison_df


def run_review_intelligence_pipeline(
    reviews_df: pd.DataFrame,
    text_col: str = 'review_text',
    id_col: str = 'review_id',
    date_col: str = 'review_date'
) -> Tuple[ReviewIntelligenceEngine, Dict]:
    """
    Convenience function to run complete review intelligence pipeline.
    
    Args:
        reviews_df: DataFrame with reviews
        text_col: Column name for review text
        id_col: Column name for review ID
        date_col: Column name for review date
    
    Returns:
        Tuple of (engine, analysis results)
    """
    engine = ReviewIntelligenceEngine()
    
    # Analyze reviews
    analysis_results = engine.batch_analyze_reviews(reviews_df, text_col, id_col)
    
    # Generate summary
    summary = engine.generate_review_summary(analysis_results)
    
    # Identify key issues
    key_issues = engine.identify_key_issues(analysis_results)
    
    # Analyze trends if date column available
    trends = None
    if date_col in reviews_df.columns:
        analysis_results[date_col] = reviews_df[date_col]
        trends = engine.analyze_sentiment_trends(analysis_results, date_col)
    
    results = {
        'analysis_results': analysis_results,
        'summary': summary,
        'key_issues': key_issues,
        'trends': trends
    }
    
    return engine, results
