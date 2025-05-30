"""
E-commerce Behavior Analysis Package

This package contains modules for analyzing e-commerce behavior data
and evaluating the impact of AI-driven recommendations.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main functions for easy access
from .user_segmentation import run_user_segmentation_analysis
from .recommendation_system import run_recommendation_analysis  
from .ab_testing import run_ab_test_analysis
from .nlp_analysis import analyze_product_reviews

__all__ = [
    'run_user_segmentation_analysis',
    'run_recommendation_analysis', 
    'run_ab_test_analysis',
    'analyze_product_reviews'
]