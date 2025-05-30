"""
Data utility functions for e-commerce behavior analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_ecommerce_data(file_path='data/ecommerce_data.csv'):
    """
    Load and perform initial validation of e-commerce dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Loaded and validated dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    print(f"Loading dataset from {file_path}...")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Basic validation
        required_columns = ['user_id', 'product_id', 'event_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
        
        # Display basic info
        print("\nDataset Info:")
        print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "- No timestamp column")
        print(f"- Unique users: {df['user_id'].nunique()}")
        print(f"- Unique products: {df['product_id'].nunique()}")
        print(f"- Event types: {df['event_type'].unique().tolist()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_dataset(df):
    """
    Clean and preprocess the e-commerce dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    print("Cleaning dataset...")
    
    # Create a copy
    cleaned_df = df.copy()
    
    # Convert timestamp to datetime
    if 'timestamp' in cleaned_df.columns:
        cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
    
    # Remove duplicates
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - len(cleaned_df)
    
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    missing_summary = cleaned_df.isnull().sum()
    if missing_summary.sum() > 0:
        print("Missing values found:")
        for col, missing_count in missing_summary[missing_summary > 0].items():
            print(f"  - {col}: {missing_count} ({missing_count/len(cleaned_df)*100:.2f}%)")
    
    # Basic data type optimization
    if 'user_id' in cleaned_df.columns:
        cleaned_df['user_id'] = cleaned_df['user_id'].astype('category')
    if 'product_id' in cleaned_df.columns:
        cleaned_df['product_id'] = cleaned_df['product_id'].astype('category')
    if 'event_type' in cleaned_df.columns:
        cleaned_df['event_type'] = cleaned_df['event_type'].astype('category')
    
    print(f"Dataset cleaned: {len(cleaned_df)} rows remaining")
    
    return cleaned_df

def save_processed_data(df, file_path='data/processed_ecommerce_data.csv'):
    """
    Save processed dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataset
    file_path : str
        Output file path
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def get_data_summary(df):
    """
    Generate a comprehensive summary of the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to summarize
    
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'date_range': None,
        'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else None,
        'unique_products': df['product_id'].nunique() if 'product_id' in df.columns else None,
        'event_types': df['event_type'].unique().tolist() if 'event_type' in df.columns else None,
        'missing_values': df.isnull().sum().to_dict()
    }
    
    if 'timestamp' in df.columns:
        summary['date_range'] = {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max(),
            'days': (df['timestamp'].max() - df['timestamp'].min()).days
        }
    
    return summary