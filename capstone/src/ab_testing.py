import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union

class ABTesting:
    def __init__(self):
        self.control_data = None
        self.treatment_data = None
        self.metrics = {}
        
    def load_data(self, control_data: pd.DataFrame, treatment_data: pd.DataFrame):
        """
        Load control and treatment data
        
        Parameters:
        -----------
        control_data : pandas.DataFrame
            DataFrame containing control group data
        treatment_data : pandas.DataFrame
            DataFrame containing treatment group data
        """
        self.control_data = control_data
        self.treatment_data = treatment_data
        
    def calculate_metrics(self, metric_columns: List[str], 
                         group_column: str = 'group',
                         user_column: str = 'user_id'):
        """
        Calculate metrics for both groups
        
        Parameters:
        -----------
        metric_columns : List[str]
            List of column names to calculate metrics for
        group_column : str
            Name of the column containing group assignment
        user_column : str
            Name of the column containing user IDs
        """
        self.metrics = {}
        
        for metric in metric_columns:
            # Calculate per-user metrics
            control_metric = self.control_data.groupby(user_column)[metric].mean()
            treatment_metric = self.treatment_data.groupby(user_column)[metric].mean()
            
            self.metrics[metric] = {
                'control': control_metric,
                'treatment': treatment_metric
            }
    
    def run_t_test(self, metric: str, alpha: float = 0.05) -> Dict:
        """
        Run t-test for a specific metric
        
        Parameters:
        -----------
        metric : str
            Name of the metric to test
        alpha : float
            Significance level
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found. Calculate metrics first.")
        
        control_values = self.metrics[metric]['control']
        treatment_values = self.metrics[metric]['treatment']
        
        # Run t-test
        t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
        
        # Calculate effect size (Cohen's d)
        control_mean = control_values.mean()
        treatment_mean = treatment_values.mean()
        control_std = control_values.std()
        treatment_std = treatment_values.std()
        
        # Pooled standard deviation
        pooled_std = np.sqrt((control_std ** 2 + treatment_std ** 2) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        return {
            'metric': metric,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'effect_size': cohens_d
        }
    
    def run_chi_square_test(self, metric: str, alpha: float = 0.05) -> Dict:
        """
        Run chi-square test for categorical metrics
        
        Parameters:
        -----------
        metric : str
            Name of the metric to test
        alpha : float
            Significance level
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found. Calculate metrics first.")
        
        # Create contingency table
        control_counts = self.metrics[metric]['control'].value_counts()
        treatment_counts = self.metrics[metric]['treatment'].value_counts()
        
        # Ensure both series have the same index
        all_categories = sorted(set(control_counts.index) | set(treatment_counts.index))
        control_counts = control_counts.reindex(all_categories, fill_value=0)
        treatment_counts = treatment_counts.reindex(all_categories, fill_value=0)
        
        # Create contingency table
        contingency_table = pd.DataFrame({
            'control': control_counts,
            'treatment': treatment_counts
        })
        
        # Run chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'metric': metric,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'significant': p_value < alpha,
            'degrees_of_freedom': dof,
            'contingency_table': contingency_table
        }
    
    def plot_metric_distribution(self, metric: str, bins: int = 30):
        """
        Plot distribution of a metric for both groups
        
        Parameters:
        -----------
        metric : str
            Name of the metric to plot
        bins : int
            Number of bins for histogram
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found. Calculate metrics first.")
        
        plt.figure(figsize=(10, 6))
        
        # Plot histograms
        sns.histplot(data=self.metrics[metric]['control'], 
                    label='Control', alpha=0.5, bins=bins)
        sns.histplot(data=self.metrics[metric]['treatment'], 
                    label='Treatment', alpha=0.5, bins=bins)
        
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Count')
        plt.legend()
        plt.show()
    
    def plot_metric_comparison(self, metric: str):
        """
        Plot box plot comparing metric between groups
        
        Parameters:
        -----------
        metric : str
            Name of the metric to plot
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found. Calculate metrics first.")
        
        # Prepare data for plotting
        control_data = pd.DataFrame({
            'value': self.metrics[metric]['control'],
            'group': 'Control'
        })
        treatment_data = pd.DataFrame({
            'value': self.metrics[metric]['treatment'],
            'group': 'Treatment'
        })
        plot_data = pd.concat([control_data, treatment_data])
        
        # Create box plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=plot_data, x='group', y='value')
        plt.title(f'Comparison of {metric}')
        plt.xlabel('Group')
        plt.ylabel(metric)
        plt.show()
    
    def calculate_sample_size(self, metric: str, 
                            mde: float, 
                            alpha: float = 0.05, 
                            power: float = 0.8) -> int:
        """
        Calculate required sample size for a given metric
        
        Parameters:
        -----------
        metric : str
            Name of the metric to calculate sample size for
        mde : float
            Minimum detectable effect (as a proportion)
        alpha : float
            Significance level
        power : float
            Desired statistical power
            
        Returns:
        --------
        int
            Required sample size per group
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found. Calculate metrics first.")
        
        # Get control group statistics
        control_mean = self.metrics[metric]['control'].mean()
        control_std = self.metrics[metric]['control'].std()
        
        # Calculate effect size
        effect_size = mde * control_mean / control_std
        
        # Calculate sample size
        n = stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power)
        n = (2 * (n ** 2)) / (effect_size ** 2)
        
        return int(np.ceil(n))
    
    def generate_report(self, metrics: List[str], alpha: float = 0.05) -> pd.DataFrame:
        """
        Generate a comprehensive report of A/B test results
        
        Parameters:
        -----------
        metrics : List[str]
            List of metrics to include in the report
        alpha : float
            Significance level
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing test results
        """
        results = []
        
        for metric in metrics:
            # Run appropriate test based on metric type
            if self.metrics[metric]['control'].dtype in ['int64', 'float64']:
                test_result = self.run_t_test(metric, alpha)
                results.append({
                    'metric': metric,
                    'test_type': 't-test',
                    'statistic': test_result['t_statistic'],
                    'p_value': test_result['p_value'],
                    'significant': test_result['significant'],
                    'control_mean': test_result['control_mean'],
                    'treatment_mean': test_result['treatment_mean'],
                    'effect_size': test_result['effect_size']
                })
            else:
                test_result = self.run_chi_square_test(metric, alpha)
                results.append({
                    'metric': metric,
                    'test_type': 'chi-square',
                    'statistic': test_result['chi2_statistic'],
                    'p_value': test_result['p_value'],
                    'significant': test_result['significant'],
                    'control_mean': None,
                    'treatment_mean': None,
                    'effect_size': None
                })
        
        return pd.DataFrame(results)
