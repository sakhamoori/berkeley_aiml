# E-commerce Behavior Analysis: Impact of AI-Driven Recommendations

## Project Overview
This project analyzes e-commerce behavior data to understand how AI-driven recommendations influence shopping behavior, user engagement, and purchase decisions.

## Dataset
- **Source**: Kaggle - E-commerce Behavior Data from Multi Category Store
- **URL**: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

## Project Structure
- `notebooks/`: Jupyter notebooks with main analysis
- `src/`: Python modules for different analysis components
- `reports/`: Final reports and visualizations
- `data/`: Data files (add your downloaded dataset here)
- `utils/`: Utility functions for data processing

## Installation

1. Clone or download this project
2. Install required packages:
   ```bash
   pip install -r requirements.txt
3. Download the dataset from Kaggle and place it in the data/ folder
Run the notebooks in order:

exploratory_analysis.ipynb - Initial data exploration
main_analysis.ipynb - Complete analysis pipeline

Files Description
Notebooks

main_analysis.ipynb: Complete analysis pipeline including EDA, clustering, recommendations, and A/B testing
exploratory_analysis.ipynb: Initial data exploration and cleaning

Source Code

user_segmentation.py: Advanced user clustering and segmentation analysis
recommendation_system.py: Multiple recommendation algorithms implementation
ab_testing.py: A/B testing simulation and analysis
nlp_analysis.py: Natural language processing for product reviews

Reports

initial_report.md: Summary report of findings and insights

Usage

Data Preparation: Place your dataset in the data/ folder
Run Analysis: Execute notebooks in the notebooks/ folder
Review Results: Check the reports/ folder for generated insights

Requirements

Python 3.8+
Jupyter Notebook
See requirements.txt for full package list