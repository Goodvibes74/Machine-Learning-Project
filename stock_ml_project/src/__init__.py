"""
Stock Prediction Package

This package contains modules for stock market prediction using
machine learning and sentiment analysis.

Modules:
- data_collection: Download stock data from Yahoo Finance
- preprocessing: Clean and validate data
- feature_engineering: Create predictive features
- models: Train and evaluate ML models
- backtesting: Walk-forward validation

Author: Machine Learning Project Group O
Institution: Makerere University
"""

__version__ = '1.0.0'
__author__ = 'Group O - Machine Learning Project'

# Import main functions for easy access
from .data_collection import (
    download_stock_data,
    download_multiple_stocks,
    load_stock_data
)

from .preprocessing import (
    preprocess_stock_data,
    clean_stock_data
)

from .feature_engineering import (
    engineer_all_features,
    prepare_ml_data
)

from .models import (
    train_random_forest,
    evaluate_model,
    train_and_evaluate_pipeline
)

from .backtesting import (
    walk_forward_validation,
    compare_strategies
)

__all__ = [
    'download_stock_data',
    'download_multiple_stocks',
    'load_stock_data',
    'preprocess_stock_data',
    'clean_stock_data',
    'engineer_all_features',
    'prepare_ml_data',
    'train_random_forest',
    'evaluate_model',
    'train_and_evaluate_pipeline',
    'walk_forward_validation',
    'compare_strategies'
]