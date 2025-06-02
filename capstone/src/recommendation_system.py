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
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix

class RecommendationSystem:
    def __init__(self, use_gpu=False, batch_size=10000, temp_dir=None, timeout=3600, top_k=10):
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
        top_k : int, default=10
            Number of top similar users to keep in the sparse similarity matrix
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
        self.start_time = None
        self.top_k = top_k
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
        self.start_time = time.time()
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
            if time.time() - self.start_time > self.timeout:
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
        """Calculate similarities using sparse top-k user similarity matrix"""
        n_items = len(self.item_mapping)
        n_users = len(self.user_mapping)
        chunk_size = 500  # For item similarity, keep as is for now
        save_chunk_size = 100
        
        # Calculate item similarities in chunks
        print("Calculating item similarities...")
        item_sim_dir = os.path.join(self.checkpoint_dir, 'item_similarity_chunks')
        os.makedirs(item_sim_dir, exist_ok=True)
        
        # Check for existing chunks
        existing_chunks = sorted([f for f in os.listdir(item_sim_dir) if f.startswith('chunk_')])
        if existing_chunks:
            print("Loading existing item similarity chunks...")
            self.item_similarity_matrix = np.zeros((n_items, n_items))
            for chunk_file in existing_chunks:
                chunk_idx = int(chunk_file.split('_')[1].split('.')[0])
                chunk_data = np.load(os.path.join(item_sim_dir, chunk_file))
                start_idx = chunk_idx * save_chunk_size
                end_idx = min(start_idx + save_chunk_size, n_items)
                self.item_similarity_matrix[start_idx:end_idx, :] = chunk_data
        else:
            self.item_similarity_matrix = np.zeros((n_items, n_items))
        
        for i in tqdm(range(0, n_items, chunk_size), desc="Processing item chunks"):
            if time.time() - self.start_time > self.timeout:
                print("Timeout reached. Saving progress...")
                # Save current chunk
                chunk_idx = i // save_chunk_size
                chunk_data = self.item_similarity_matrix[i:i+save_chunk_size, :]
                np.save(os.path.join(item_sim_dir, f'chunk_{chunk_idx}.npy'), chunk_data)
                raise TimeoutError("Item similarity calculation timeout reached")
                
            end_i = min(i + chunk_size, n_items)
            chunk = self.user_item_matrix[:, i:end_i]
            
            # Calculate similarities for this chunk
            chunk_similarities = cosine_similarity(chunk.T, self.user_item_matrix.T)
            self.item_similarity_matrix[i:end_i, :] = chunk_similarities
            
            # Save completed chunks
            for save_idx in range(i, end_i, save_chunk_size):
                save_end = min(save_idx + save_chunk_size, end_i)
                if save_end - save_idx == save_chunk_size:  # Only save complete chunks
                    chunk_idx = save_idx // save_chunk_size
                    chunk_data = self.item_similarity_matrix[save_idx:save_end, :]
                    np.save(os.path.join(item_sim_dir, f'chunk_{chunk_idx}.npy'), chunk_data)
            
            # Clear memory
            del chunk_similarities
            gc.collect()
            
            # Save progress periodically
            if i % (chunk_size * 5) == 0:
                print(f"Memory usage during item similarity calculation: {self._get_memory_usage():.2f} GB")
        
        # Calculate user similarities using sparse top-k approach
        print("Calculating user similarities (sparse top-k)...")
        self.user_similarity_matrix = compute_topk_user_similarities(self.user_item_matrix, k=self.top_k)
        print(f"User similarity matrix shape: {self.user_similarity_matrix.shape}, nnz: {self.user_similarity_matrix.nnz}")
        # Optionally, save to disk if needed
        # from scipy.sparse import save_npz
        # save_npz(os.path.join(self.checkpoint_dir, 'user_similarity_sparse.npz'), self.user_similarity_matrix)
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=5, item_weight=0.5):
        """Get hybrid recommendations with memory-efficient processing (sparse user similarity)"""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        user_items = self.user_item_matrix[user_idx].nonzero()[1]
        
        # Get item-based recommendations
        item_scores = self.item_similarity_matrix[user_items].mean(axis=0)
        item_scores[user_items] = -np.inf
        top_items = np.argsort(item_scores)[-n_recommendations:][::-1]
        item_recs = [self.reverse_item_mapping[item] for item in top_items]
        
        # Get user-based recommendations (sparse)
        similar_users = self.user_similarity_matrix.getrow(user_idx).toarray().flatten()
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

def compute_topk_user_similarities(user_item_matrix, k=10):
    # Fit NearestNeighbors on user vectors
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='auto')
    nn.fit(user_item_matrix)
    distances, indices = nn.kneighbors(user_item_matrix)
    
    n_users = user_item_matrix.shape[0]
    user_sim_matrix = lil_matrix((n_users, n_users))
    
    for i in range(n_users):
        for j, dist in zip(indices[i][1:], distances[i][1:]):  # skip self (first neighbor)
            user_sim_matrix[i, j] = 1 - dist  # similarity = 1 - cosine distance
    
    return user_sim_matrix.tocsr()