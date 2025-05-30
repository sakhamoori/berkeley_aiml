import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class RecommendationSystem:
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        
    def fit(self, df):
        """
        Fit the recommendation system to the data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing user-item interactions with columns:
            - user_id
            - product_id
            - event_type (optional)
        """
        # Create user-item interaction matrix
        self.user_item_matrix = pd.crosstab(df['user_id'], df['product_id'])
        
        # Convert to sparse matrix for efficiency
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Calculate item similarity matrix
        self.item_similarity_matrix = cosine_similarity(sparse_matrix.T)
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        # Calculate user similarity matrix
        self.user_similarity_matrix = cosine_similarity(sparse_matrix)
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
    
    def get_item_recommendations(self, user_id, n_recommendations=5):
        """
        Get item-based collaborative filtering recommendations for a user
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user to get recommendations for
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        list
            List of recommended product IDs
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's items
        user_items = self.user_item_matrix.loc[user_id]
        user_items = user_items[user_items > 0].index
        
        if len(user_items) == 0:
            return []
        
        # Get similar items
        similar_items = self.item_similarity_matrix[user_items].mean().sort_values(ascending=False)
        similar_items = similar_items[~similar_items.index.isin(user_items)]
        
        return similar_items.head(n_recommendations).index.tolist()
    
    def get_user_recommendations(self, user_id, n_recommendations=5):
        """
        Get user-based collaborative filtering recommendations
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user to get recommendations for
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        list
            List of recommended product IDs
        """
        if user_id not in self.user_similarity_matrix.index:
            return []
        
        # Get similar users
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)
        similar_users = similar_users[similar_users.index != user_id]
        
        # Get items from similar users
        similar_user_items = self.user_item_matrix.loc[similar_users.index]
        user_items = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        
        # Calculate item scores
        item_scores = {}
        for item in similar_user_items.columns:
            if item not in user_items:
                score = 0
                for user in similar_users.index:
                    score += similar_users[user] * similar_user_items.loc[user, item]
                item_scores[item] = score
        
        # Sort and return top recommendations
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in recommendations[:n_recommendations]]
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=5, item_weight=0.5):
        """
        Get hybrid recommendations combining item-based and user-based approaches
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user to get recommendations for
        n_recommendations : int
            Number of recommendations to return
        item_weight : float
            Weight given to item-based recommendations (0-1)
            
        Returns:
        --------
        list
            List of recommended product IDs
        """
        item_recs = self.get_item_recommendations(user_id, n_recommendations * 2)
        user_recs = self.get_user_recommendations(user_id, n_recommendations * 2)
        
        # Combine recommendations with weights
        combined_scores = {}
        for item in set(item_recs + user_recs):
            item_score = item_weight * (1 - item_recs.index(item) / len(item_recs)) if item in item_recs else 0
            user_score = (1 - item_weight) * (1 - user_recs.index(item) / len(user_recs)) if item in user_recs else 0
            combined_scores[item] = item_score + user_score
        
        # Sort and return top recommendations
        recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in recommendations[:n_recommendations]]
    
    def evaluate_recommendations(self, test_data, n_recommendations=5):
        """
        Evaluate recommendation system performance
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            Test dataset with user-item interactions
        n_recommendations : int
            Number of recommendations to evaluate
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Create test user-item matrix
        test_matrix = pd.crosstab(test_data['user_id'], test_data['product_id'])
        
        # Calculate metrics
        precision = []
        recall = []
        
        for user_id in test_matrix.index:
            if user_id in self.user_item_matrix.index:
                # Get recommendations
                recommendations = self.get_hybrid_recommendations(user_id, n_recommendations)
                
                # Get actual items
                actual_items = set(test_matrix.loc[user_id][test_matrix.loc[user_id] > 0].index)
                
                # Calculate precision and recall
                if len(recommendations) > 0:
                    hits = len(set(recommendations) & actual_items)
                    precision.append(hits / len(recommendations))
                    recall.append(hits / len(actual_items) if len(actual_items) > 0 else 0)
        
        return {
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1_score': 2 * (np.mean(precision) * np.mean(recall)) / (np.mean(precision) + np.mean(recall))
        }
