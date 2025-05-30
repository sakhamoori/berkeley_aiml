import os
import kaggle
import pandas as pd
from pathlib import Path

def download_dataset():
    """
    Download the e-commerce behavior dataset from Kaggle
    """
    # Create data directory if it doesn't exist
    data_dir = Path('../data')
    data_dir.mkdir(exist_ok=True)
    
    # Download the dataset
    kaggle.api.dataset_download_files(
        'mkechinov/ecommerce-behavior-data-from-multi-category-store',
        path=str(data_dir),
        unzip=True
    )
    
    print("Dataset downloaded successfully!")

if __name__ == "__main__":
    download_dataset() 