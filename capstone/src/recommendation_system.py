import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from tqdm.notebook import tqdm
import gc
import os
import psutil
import tempfile
import time
import pickle
from pathlib import Path

class RecommendationSystem:
    def __init__(self, use_gpu=False, batch_size=10000, temp_dir=None, timeout=3600):
        """
        Initialize the recommendation system
        
        Parameters:
        -----------
        use_gpu : bool, default=False
            Whether to use GPU acceleration (currently disabled)
        batch_size : int, default=10000
            Size of batches for processing large datasets
        temp_dir : str, optional
            Directory to store temporary files. If None, uses system temp directory
        timeout : int, default=3600
            Maximum time in seconds to process each major step
        """
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        self.use_gpu = False  # Force CPU mode
        self.batch_size = batch_size
        self.temp_dir = tempfile.mkdtemp() if temp_dir is None else temp_dir
        self.timeout = timeout
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def _get_memory_usage(self):
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
        
    def _create_mappings(self, df):
        """Create mappings for users and items to reduce memory usage"""
        print("Creating user mappings...")
        self.user_mapping = {user: idx for idx, user in enumerate(df['user_id'].unique())}
        print("Creating item mappings...")
        self.item_mapping = {item: idx for idx, item in enumerate(df['product_id'].unique())}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Save mappings
        with open(os.path.join(self.checkpoint_dir, 'mappings.pkl'), 'wb') as f:
            pickle.dump({
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_user_mapping': self.reverse_user_mapping,
                'reverse_item_mapping': self.reverse_item_mapping
            }, f)
        
    def _process_batch(self, batch_df):
        """Process a batch of data"""
        rows = [self.user_mapping[user] for user in batch_df['user_id']]
        cols = [self.item_mapping[item] for item in batch_df['product_id']]
        data = np.ones(len(rows))
        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_mapping), len(self.item_mapping))
        )
        
    def fit(self, df):
        """Fit the recommendation system using batch processing"""
        start_time = time.time()
        print(f"Initial memory usage: {self._get_memory_usage():.2f} GB")
        
        # Check for existing checkpoints
        if os.path.exists(os.path.join(self.checkpoint_dir, 'mappings.pkl')):
            print("Loading existing mappings...")
            with open(os.path.join(self.checkpoint_dir, 'mappings.pkl'), 'rb') as f:
                mappings = pickle.load(f)
                self.user_mapping = mappings['user_mapping']
                self.item_mapping = mappings['item_mapping']
                self.reverse_user_mapping = mappings['reverse_user_mapping']
                self.reverse_item_mapping = mappings['reverse_item_mapping']
        else:
            print("Creating mappings...")
            self._create_mappings(df)
        
        print("Processing data in batches...")
        n_batches = len(df) // self.batch_size + (1 if len(df) % self.batch_size else 0)
        
        # Initialize or load user-item matrix
        matrix_path = os.path.join(self.checkpoint_dir, 'user_item_matrix.npz')
        if os.path.exists(matrix_path):
            print("Loading existing user-item matrix...")
            self.user_item_matrix = load_npz(matrix_path)
        else:
            self.user_item_matrix = csr_matrix((len(self.user_mapping), len(self.item_mapping)))
        
        # Process batches
        for i in tqdm(range(n_batches), desc="Processing batches"):
            if time.time() - start_time > self.timeout:
                print("Timeout reached. Saving progress...")
                save_npz(matrix_path, self.user_item_matrix)
                raise TimeoutError("Processing timeout reached")
                
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # Process batch
            batch_matrix = self._process_batch(batch_df)
            self.user_item_matrix += batch_matrix
            
            # Clear memory
            del batch_matrix
            gc.collect()
            
            # Save progress every 10 batches
            if i % 10 == 0:
                print(f"Memory usage after batch {i}: {self._get_memory_usage():.2f} GB")
                save_npz(matrix_path, self.user_item_matrix)
        
        print(f"Memory usage after creating user-item matrix: {self._get_memory_usage():.2f} GB")
        print("Calculating similarity matrices...")
        self._calculate_similarities_chunked()
        
    def _calculate_similarities_chunked(self):
        """Calculate similarities using chunked processing"""
        n_items = len(self.item_mapping)
        n_users = len(self.user_mapping)
        chunk_size = 500  # Reduced chunk size for better memory management
        
        # Calculate item similarities in chunks
        print("Calculating item similarities...")
        item_sim_path = os.path.join(self.checkpoint_dir, 'item_similarity.npy')
        
        if os.path.exists(item_sim_path):
            print("Loading existing item similarity matrix...")
            self.item_similarity_matrix = np.load(item_sim_path)
        else:
            self.item_similarity_matrix = np.zeros((n_items, n_items))
        
        for i in tqdm(range(0, n_items, chunk_size), desc="Processing item chunks"):
            if time.time() - start_time > self.timeout:
                print("Timeout reached. Saving progress...")
                np.save(item_sim_path, self.item_similarity_matrix)
                raise TimeoutError("Item similarity calculation timeout reached")
                
            end_i = min(i + chunk_size, n_items)
            chunk = self.user_item_matrix[:, i:end_i]
            
            # Calculate similarities for this chunk
            chunk_similarities = cosine_similarity(chunk.T, self.user_item_matrix.T)
            self.item_similarity_matrix[i:end_i, :] = chunk_similarities
            
            # Clear memory
            del chunk_similarities
            gc.collect()
            
            # Save progress periodically
            if i % (chunk_size * 5) == 0:
                print(f"Memory usage during item similarity calculation: {self._get_memory_usage():.2f} GB")
                np.save(item_sim_path, self.item_similarity_matrix)
        
        # Calculate user similarities in chunks
        print("Calculating user similarities...")
        user_sim_path = os.path.join(self.checkpoint_dir, 'user_similarity.npy')
        
        if os.path.exists(user_sim_path):
            print("Loading existing user similarity matrix...")
            self.user_similarity_matrix = np.load(user_sim_path)
        else:
            self.user_similarity_matrix = np.zeros((n_users, n_users))
        
        for i in tqdm(range(0, n_users, chunk_size), desc="Processing user chunks"):
            if time.time() - start_time > self.timeout:
                print("Timeout reached. Saving progress...")
                np.save(user_sim_path, self.user_similarity_matrix)
                raise TimeoutError("User similarity calculation timeout reached")
                
            end_i = min(i + chunk_size, n_users)
            chunk = self.user_item_matrix[i:end_i, :]
            
            # Calculate similarities for this chunk
            chunk_similarities = cosine_similarity(chunk, self.user_item_matrix)
            self.user_similarity_matrix[i:end_i, :] = chunk_similarities
            
            # Clear memory
            del chunk_similarities
            gc.collect()
            
            # Save progress periodically
            if i % (chunk_size * 5) == 0:
                print(f"Memory usage during user similarity calculation: {self._get_memory_usage():.2f} GB")
                np.save(user_sim_path, self.user_similarity_matrix)
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=5, item_weight=0.5):
        """Get hybrid recommendations with memory-efficient processing"""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        user_items = self.user_item_matrix[user_idx].nonzero()[1]
        
        # Get item-based recommendations
        item_scores = self.item_similarity_matrix[user_items].mean(axis=0)
        item_scores[user_items] = -np.inf
        top_items = np.argsort(item_scores)[-n_recommendations:][::-1]
        item_recs = [self.reverse_item_mapping[item] for item in top_items]
        
        # Get user-based recommendations
        similar_users = self.user_similarity_matrix[user_idx]
        similar_users[user_idx] = -np.inf
        top_users = np.argsort(similar_users)[-n_recommendations:][::-1]
        
        # Get items from similar users
        similar_user_items = set()
        for user in top_users:
            items = self.user_item_matrix[user].nonzero()[1]
            similar_user_items.update(items)
        
        # Remove items user has already interacted with
        similar_user_items = similar_user_items - set(user_items)
        
        # Calculate item scores
        item_scores = {}
        for item in similar_user_items:
            score = 0
            for user in top_users:
                score += similar_users[user] * self.user_item_matrix[user, item]
            item_scores[self.reverse_item_mapping[item]] = score
        
        # Sort and return top recommendations
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in recommendations[:n_recommendations]]