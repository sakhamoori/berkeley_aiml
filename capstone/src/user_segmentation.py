import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class UserSegmentation:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_features = None
        self.cluster_centers = None
        
    def _calculate_rfm(self, df):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each user
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing user interactions with columns:
            - user_id
            - event_time
            - event_type
            - price (optional)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with RFM metrics for each user
        """
        # Convert event_time to datetime if it's not already
        df['event_time'] = pd.to_datetime(df['event_time'])
        
        # Calculate recency (days since last interaction)
        recency = df.groupby('user_id')['event_time'].max()
        recency = (df['event_time'].max() - recency).dt.days
        
        # Calculate frequency (number of interactions)
        frequency = df.groupby('user_id').size()
        
        # Calculate monetary value (if price is available)
        if 'price' in df.columns:
            monetary = df.groupby('user_id')['price'].sum()
        else:
            monetary = frequency  # Use frequency as proxy if price not available
        
        # Combine metrics
        rfm = pd.DataFrame({
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary
        })
        
        return rfm
    
    def _calculate_behavior_features(self, df):
        """
        Calculate additional behavior features for each user
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing user interactions
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with behavior features for each user
        """
        # Calculate number of unique products viewed
        unique_products = df.groupby('user_id')['product_id'].nunique()
        
        # Calculate number of unique categories viewed
        if 'category_code' in df.columns:
            unique_categories = df.groupby('user_id')['category_code'].nunique()
        else:
            unique_categories = pd.Series(0, index=unique_products.index)
        
        # Calculate session count
        if 'user_session' in df.columns:
            session_count = df.groupby('user_id')['user_session'].nunique()
        else:
            session_count = pd.Series(0, index=unique_products.index)
        
        # Combine features
        behavior_features = pd.DataFrame({
            'unique_products': unique_products,
            'unique_categories': unique_categories,
            'session_count': session_count
        })
        
        return behavior_features
    
    def fit(self, df):
        """
        Fit the segmentation model to the data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing user interactions
        """
        # Calculate RFM metrics
        rfm = self._calculate_rfm(df)
        
        # Calculate behavior features
        behavior_features = self._calculate_behavior_features(df)
        
        # Combine all features
        self.user_features = pd.concat([rfm, behavior_features], axis=1)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(self.user_features)
        
        # Fit clustering model
        self.kmeans.fit(scaled_features)
        
        # Store cluster centers
        self.cluster_centers = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_),
            columns=self.user_features.columns
        )
        
        # Add cluster labels to user features
        self.user_features['cluster'] = self.kmeans.labels_
        
        return self
    
    def get_cluster_characteristics(self):
        """
        Get characteristics of each cluster
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cluster characteristics
        """
        if self.user_features is None:
            raise ValueError("Model must be fit before getting cluster characteristics")
        
        # Calculate mean values for each feature by cluster
        cluster_means = self.user_features.groupby('cluster').mean()
        
        # Calculate cluster sizes
        cluster_sizes = self.user_features['cluster'].value_counts()
        
        # Add cluster sizes to characteristics
        cluster_means['size'] = cluster_sizes
        
        return cluster_means
    
    def predict(self, df):
        """
        Predict cluster for new users
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing user interactions
            
        Returns:
        --------
        pandas.Series
            Cluster labels for each user
        """
        # Calculate features for new users
        rfm = self._calculate_rfm(df)
        behavior_features = self._calculate_behavior_features(df)
        user_features = pd.concat([rfm, behavior_features], axis=1)
        
        # Scale features
        scaled_features = self.scaler.transform(user_features)
        
        # Predict clusters
        clusters = self.kmeans.predict(scaled_features)
        
        return pd.Series(clusters, index=user_features.index)
    
    def plot_cluster_characteristics(self, features=None):
        """
        Plot characteristics of each cluster
        
        Parameters:
        -----------
        features : list
            List of features to plot (default: all features)
        """
        if self.user_features is None:
            raise ValueError("Model must be fit before plotting cluster characteristics")
        
        if features is None:
            features = self.user_features.columns.drop('cluster')
        
        # Create subplots
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 5 * n_features))
        
        # Plot each feature
        for i, feature in enumerate(features):
            sns.boxplot(data=self.user_features, x='cluster', y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} by Cluster')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_clustering(self):
        """
        Evaluate clustering quality using silhouette score
        
        Returns:
        --------
        float
            Silhouette score
        """
        if self.user_features is None:
            raise ValueError("Model must be fit before evaluating clustering")
        
        # Scale features
        scaled_features = self.scaler.transform(self.user_features.drop('cluster', axis=1))
        
        # Calculate silhouette score
        score = silhouette_score(scaled_features, self.kmeans.labels_)
        
        return score
