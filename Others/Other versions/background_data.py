"""
Background Data Generation System
Generates and stores background data for time series analysis
Supports: Random Sampling, K-Means, Frequency/Wavelet Clustering, Matrix Profile
"""

import numpy as np
import pandas as pd
import sqlite3
import time
import json
import os
import psutil
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch

# Signal Processing
from scipy.stats import spearmanr
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import cdist, euclidean
import pywt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Optional libraries with graceful fallback
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("⚠️  dtaidistance not available - using fallback")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("⚠️  ruptures not available - using fallback")

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['NUMBA_DISABLE_CUDA'] = '1'
    import stumpy
    STUMPY_AVAILABLE = True
    print("✅ stumpy loaded (CPU-only mode)")
except (ImportError, OSError) as e:
    STUMPY_AVAILABLE = False
    print(f"⚠️  stumpy not available - using fallback (error: {type(e).__name__})")


# ============================
# CONFIGURATION - MASTER CONTROL
# ============================

# Background generation methods control
ENABLE_RANDOM_SAMPLE = True
ENABLE_KMEANS = True
ENABLE_FREQUENCY_CLUSTERING = True
ENABLE_WAVELET_CLUSTERING = True
ENABLE_MATRIX_PROFILE = True
ENABLE_FEATURE_MEAN = True

# Cluster optimization settings
CLUSTER_OPTIMIZATION_ENABLED = True
CLUSTER_RANGE = range(2, 21)
CLUSTER_SAMPLE_PCT = 0.10

# Sample counts for non-clustering methods
RANDOM_SAMPLE_COUNTS = [10, 50, 100]

# Matrix Profile configurations
MATRIX_PROFILE_SAMPLE_COUNTS = [10, 50, 100]  # List of n_samples
MATRIX_PROFILE_WINDOW_SIZES = [6, 12, 24]     # List of window_sizes

# General settings
N_SAMPLES_PER_CONFIG = 10
RANDOM_SEED = 42
MEMORY_THRESHOLD_PCT = 85
MEMORY_CHECK_INTERVAL = 10

# Database paths
BENCHMARK_DB = "benchmark_results.db"
BACKGROUND_DB = "background_data.db"
RESULTS_BASE_DIR = "results"


# ============================
# CONFIGURATION FILTER FUNCTION
# ============================

def is_background_enabled(bg_type, bg_params):
    """Check if a background type is enabled by configuration"""
    if bg_type == 'random_sample':
        if not ENABLE_RANDOM_SAMPLE:
            return False
        n_samples = bg_params.get('n_samples', 0)
        return n_samples in RANDOM_SAMPLE_COUNTS
    
    elif bg_type == 'kmeans':
        return ENABLE_KMEANS
    
    elif bg_type == 'frequency_clustering':
        return ENABLE_FREQUENCY_CLUSTERING
    
    elif bg_type == 'wavelet_clustering':
        return ENABLE_WAVELET_CLUSTERING
    
    elif bg_type == 'matrix_profile_sampling':
        if not ENABLE_MATRIX_PROFILE:
            return False
        n_samples = bg_params.get('n_samples', 0)
        window_size = bg_params.get('window_size', 0)
        return (n_samples in MATRIX_PROFILE_SAMPLE_COUNTS and 
                window_size in MATRIX_PROFILE_WINDOW_SIZES)
    
    elif bg_type == 'feature_mean':
        return ENABLE_FEATURE_MEAN
    
    else:
        return False


# ============================
# DATABASE INITIALIZATION
# ============================

def init_background_database(db_path=BACKGROUND_DB):
    """Initialize background data database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_sample_indices (
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            test_indices_json TEXT NOT NULL,
            n_samples INTEGER NOT NULL,
            random_seed INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (primary_use, option_number)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS background_data (
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            background_type TEXT NOT NULL,
            background_params TEXT NOT NULL,
            background_data_json TEXT NOT NULL,
            test_indices_json TEXT NOT NULL,
            background_size INTEGER NOT NULL,
            generation_time REAL NOT NULL,
            shape_info TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (primary_use, option_number, background_type, background_params)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cluster_optimization (
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            clustering_type TEXT NOT NULL,
            k INTEGER NOT NULL,
            silhouette_score REAL,
            davies_bouldin_index REAL,
            calinski_harabasz_score REAL,
            dunn_index REAL,
            is_recommended INTEGER,
            n_samples INTEGER,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (primary_use, option_number, clustering_type, k)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Background database initialized: {db_path}")


# ============================
# CLUSTER OPTIMIZATION FUNCTIONS
# ============================

def compute_dunn_index(X, labels):
    """Compute Dunn Index for clustering quality"""
    try:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return 0.0
        
        centers = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
        inter_cluster_dists = cdist(centers, centers, metric='euclidean')
        np.fill_diagonal(inter_cluster_dists, np.inf)
        min_inter_cluster = np.min(inter_cluster_dists)
        
        max_intra_cluster = 0.0
        for k in unique_labels:
            cluster_points = X[labels == k]
            if len(cluster_points) > 1:
                cluster_dists = cdist(cluster_points, cluster_points, metric='euclidean')
                max_intra_cluster = max(max_intra_cluster, np.max(cluster_dists))
        
        if max_intra_cluster == 0:
            return 0.0
        
        return min_inter_cluster / max_intra_cluster
    
    except Exception as e:
        print(f"      Warning: Dunn index computation failed: {e}")
        return 0.0


def evaluate_clustering_quality(X, k, method_name):
    """Evaluate clustering quality for a given k using 4 metrics"""
    try:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X)
        
        if len(np.unique(labels)) < 2:
            return None
        
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        dunn = compute_dunn_index(X, labels)
        
        return {
            'k': k,
            'method': method_name,
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(davies_bouldin),
            'calinski_harabasz_score': float(calinski_harabasz),
            'dunn_index': float(dunn),
            'n_samples': len(X)
        }
    
    except Exception as e:
        print(f"      Warning: Clustering evaluation failed for k={k}: {e}")
        return None


def find_optimal_clusters(X_train, clustering_type, primary_use, option_number):
    """Find optimal number of clusters using 4 metrics"""
    print(f"\n      🔍 Optimizing clusters for {clustering_type}...")
    
    n_samples = int(len(X_train) * CLUSTER_SAMPLE_PCT)
    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(len(X_train), min(n_samples, len(X_train)), replace=False)
    X_sample = X_train[sample_indices]
    
    print(f"      Using {len(X_sample)} samples ({CLUSTER_SAMPLE_PCT*100:.0f}% of training data)")
    
    if clustering_type == 'kmeans':
        features = X_sample.reshape(len(X_sample), -1)
        method_name = 'kmeans'
    
    elif clustering_type == 'frequency_clustering':
        print(f"      Extracting frequency features...")
        features = []
        for i, sample in enumerate(X_sample):
            if i % 500 == 0 and i > 0:
                print(f"        Progress: {i}/{len(X_sample)}")
            features.append(extract_signal_features(sample))
        features = np.array(features)
        method_name = 'frequency_clustering'
    
    elif clustering_type == 'wavelet_clustering':
        print(f"      Extracting wavelet features...")
        features = []
        for i, sample in enumerate(X_sample):
            if i % 500 == 0 and i > 0:
                print(f"        Progress: {i}/{len(X_sample)}")
            try:
                wav_feat = compute_wavelet_features(sample)
                features.append(wav_feat)
            except Exception as e:
                features.append(np.zeros(20))
        features = np.array(features)
        method_name = 'wavelet_clustering'
    
    else:
        print(f"      ⚠️  Unknown clustering type: {clustering_type}")
        return None
    
    valid_mask = np.all(np.isfinite(features), axis=1)
    if not np.all(valid_mask):
        print(f"      Warning: {np.sum(~valid_mask)} samples have invalid features")
        features = features[valid_mask]
    
    if len(features) < 2:
        print(f"      ❌ Not enough valid samples for clustering")
        return None
    
    print(f"      Testing k from {min(CLUSTER_RANGE)} to {max(CLUSTER_RANGE)}...")
    results = []
    
    for k in CLUSTER_RANGE:
        if k > len(features):
            break
        
        result = evaluate_clustering_quality(features, k, method_name)
        if result is not None:
            results.append(result)
    
    if len(results) == 0:
        print(f"      ❌ No valid clustering results")
        return None
    
    results_df = pd.DataFrame(results)
    
    best_silhouette_k = results_df.loc[results_df['silhouette_score'].idxmax(), 'k']
    best_davies_bouldin_k = results_df.loc[results_df['davies_bouldin_index'].idxmin(), 'k']
    best_calinski_harabasz_k = results_df.loc[results_df['calinski_harabasz_score'].idxmax(), 'k']
    best_dunn_k = results_df.loc[results_df['dunn_index'].idxmax(), 'k']
    
    recommended_k = sorted(set([best_silhouette_k, best_davies_bouldin_k, 
                                best_calinski_harabasz_k, best_dunn_k]))
    
    print(f"      ✅ Optimal clusters found:")
    print(f"         Silhouette Score: k={int(best_silhouette_k)}")
    print(f"         Davies-Bouldin Index: k={int(best_davies_bouldin_k)}")
    print(f"         Calinski-Harabasz Score: k={int(best_calinski_harabasz_k)}")
    print(f"         Dunn Index: k={int(best_dunn_k)}")
    print(f"      📊 Recommended k values: {recommended_k}")
    
    save_cluster_optimization_results(
        primary_use, option_number, clustering_type,
        results_df, recommended_k
    )
    
    return recommended_k


def save_cluster_optimization_results(primary_use, option_number, clustering_type, 
                                       results_df, recommended_k):
    """Save cluster optimization results to database"""
    conn = sqlite3.connect(BACKGROUND_DB)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    for _, row in results_df.iterrows():
        is_recommended = 1 if row['k'] in recommended_k else 0
        cursor.execute('''
            INSERT OR REPLACE INTO cluster_optimization
            (primary_use, option_number, clustering_type, k, 
             silhouette_score, davies_bouldin_index, calinski_harabasz_score, dunn_index,
             is_recommended, n_samples, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            primary_use, option_number, clustering_type, int(row['k']),
            row['silhouette_score'], row['davies_bouldin_index'],
            row['calinski_harabasz_score'], row['dunn_index'],
            is_recommended, int(row['n_samples']), timestamp
        ))
    
    conn.commit()
    conn.close()


def load_optimal_clusters(primary_use, option_number, clustering_type):
    """Load recommended cluster numbers from database"""
    try:
        conn = sqlite3.connect(BACKGROUND_DB)
        query = '''
            SELECT k FROM cluster_optimization
            WHERE primary_use = ? AND option_number = ? 
            AND clustering_type = ? AND is_recommended = 1
            ORDER BY k
        '''
        cursor = conn.cursor()
        cursor.execute(query, (primary_use, option_number, clustering_type))
        results = cursor.fetchall()
        conn.close()
        
        if len(results) > 0:
            return [int(r[0]) for r in results]
        else:
            return None
    
    except Exception as e:
        print(f"      Warning: Could not load optimal clusters: {e}")
        return None


# ============================
# SIGNAL PROCESSING FUNCTIONS
# ============================

def compute_fourier_features(signal_data):
    """Extract frequency domain features"""
    try:
        flat_signal = signal_data.flatten()
        if len(flat_signal) < 20:
            return np.zeros(10)
        fft_vals = fft(flat_signal)
        power_spectrum = np.abs(fft_vals) ** 2
        n_top = min(10, len(power_spectrum))
        top_freqs = np.sort(power_spectrum)[-n_top:]
        if len(top_freqs) < 10:
            top_freqs = np.pad(top_freqs, (0, 10 - len(top_freqs)), 'constant')
        total = np.sum(top_freqs)
        if total > 1e-10:
            return top_freqs / total
        else:
            return np.zeros(10)
    except Exception as e:
        print(f"      Warning: Fourier feature extraction failed: {e}")
        return np.zeros(10)


def compute_wavelet_features(signal_data, wavelet='db4', level=3):
    """Extract wavelet decomposition features"""
    try:
        flat_signal = signal_data.flatten()
        min_len = 2 ** (level + 1)
        if len(flat_signal) < min_len:
            flat_signal = np.pad(flat_signal, (0, min_len - len(flat_signal)), 'edge')
        coeffs = pywt.wavedec(flat_signal, wavelet, level=level)
        features = []
        for coeff in coeffs:
            if len(coeff) > 0:
                features.extend([np.mean(np.abs(coeff)), np.std(coeff), np.max(np.abs(coeff))])
            else:
                features.extend([0.0, 0.0, 0.0])
        return np.array(features)
    except Exception as e:
        print(f"      Warning: Wavelet feature extraction failed: {e}")
        return np.zeros(20)


def compute_matrix_profile(signal_data, window_size=10):
    """Compute matrix profile for pattern discovery"""
    if not STUMPY_AVAILABLE:
        return None
    try:
        flat_signal = signal_data.flatten()
        if len(flat_signal) < window_size * 2:
            print(f"      Signal too short for matrix profile (len={len(flat_signal)}, window={window_size})")
            return None
        mp = stumpy.stump(flat_signal, m=window_size)
        return mp[:, 0]
    except Exception as e:
        print(f"      Warning: Matrix profile computation failed: {e}")
        return None


def extract_signal_features(signal_data):
    """Extract comprehensive signal features for clustering"""
    try:
        features = []
        features.extend([np.mean(signal_data), np.std(signal_data), np.min(signal_data), 
                        np.max(signal_data), np.median(signal_data)])
        freq_features = compute_fourier_features(signal_data)
        features.extend(freq_features)
        wav_features = compute_wavelet_features(signal_data)
        features.extend(wav_features[:10])
        return np.array(features)
    except Exception as e:
        print(f"      Warning: Feature extraction failed: {e}")
        return np.zeros(25)


# ============================
# TEST SAMPLE INDICES MANAGEMENT
# ============================

def get_or_generate_test_indices(primary_use, option_number, n_test_samples, db_path=BACKGROUND_DB):
    """Get test indices from database if they exist, otherwise generate and save them"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT test_indices_json, n_samples, random_seed, timestamp
        FROM test_sample_indices
        WHERE primary_use = ? AND option_number = ?
    ''', (primary_use, option_number))
    
    result = cursor.fetchone()
    
    if result is not None:
        test_indices_json, n_samples, random_seed, timestamp = result
        test_indices = json.loads(test_indices_json)
        
        print(f"  📌 Using EXISTING test indices (generated: {timestamp[:10]})")
        print(f"     Indices: {test_indices}")
        print(f"     Random seed: {random_seed}, N samples: {n_samples}")
        
        conn.close()
        return test_indices
    
    else:
        print(f"  🎲 Generating NEW test indices (will be saved permanently)")
        
        np.random.seed(RANDOM_SEED)
        test_indices = np.random.choice(
            n_test_samples, 
            size=min(N_SAMPLES_PER_CONFIG, n_test_samples), 
            replace=False
        ).tolist()
        
        timestamp = datetime.now().isoformat()
        test_indices_json = json.dumps(test_indices)
        
        cursor.execute('''
            INSERT INTO test_sample_indices
            (primary_use, option_number, test_indices_json, n_samples, random_seed, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (primary_use, option_number, test_indices_json, len(test_indices), RANDOM_SEED, timestamp))
        
        conn.commit()
        
        print(f"  ✅ Test indices saved permanently to database")
        print(f"     Indices: {test_indices}")
        print(f"     Random seed: {RANDOM_SEED}, N samples: {len(test_indices)}")
        
        conn.close()
        return test_indices


# ============================
# BACKGROUND DATA GENERATION
# ============================

def generate_background_data(X_train, bg_type, bg_params):
    """Generate background data based on selected strategy"""
    np.random.seed(RANDOM_SEED)
    
    try:
        if bg_type == 'random_sample':
            n_samples = int(bg_params.get('n_samples', 10))
            n_samples = min(n_samples, len(X_train))
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            background = X_train[indices]
            print(f"      ✓ Random sample: {len(background)} samples")
            
        elif bg_type == 'kmeans':
            n_clusters = int(bg_params.get('n_clusters', 10))
            n_clusters = min(n_clusters, len(X_train))
            X_flat = X_train.reshape(len(X_train), -1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
            kmeans.fit(X_flat)
            centroids = kmeans.cluster_centers_.reshape(n_clusters, X_train.shape[1], X_train.shape[2])
            background = centroids
            print(f"      ✓ K-Means: {len(background)} centroids")
            
        elif bg_type == 'frequency_clustering':
            n_clusters = int(bg_params.get('n_clusters', 10))
            n_clusters = min(n_clusters, len(X_train))
            print(f"      Computing frequency features for {len(X_train)} samples...")
            features = []
            for i, sample in enumerate(X_train):
                if i % 500 == 0 and i > 0:
                    print(f"        Progress: {i}/{len(X_train)}")
                features.append(extract_signal_features(sample))
            features = np.array(features)
            print(f"      Features shape: {features.shape}")
            valid_mask = np.all(np.isfinite(features), axis=1)
            if not np.all(valid_mask):
                print(f"      Warning: {np.sum(~valid_mask)} samples have invalid features")
                features = features[valid_mask]
                X_train_valid = X_train[valid_mask]
            else:
                X_train_valid = X_train
            if len(features) < n_clusters:
                print(f"      Warning: Not enough valid samples ({len(features)}), reducing clusters to {len(features)}")
                n_clusters = len(features)
            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
            labels = kmeans.fit_predict(features)
            centroids_list = []
            for k in range(n_clusters):
                cluster_indices = np.where(labels == k)[0]
                if len(cluster_indices) > 0:
                    cluster_features = features[cluster_indices]
                    centroid = kmeans.cluster_centers_[k]
                    distances = np.linalg.norm(cluster_features - centroid, axis=1)
                    closest_idx = cluster_indices[np.argmin(distances)]
                    centroids_list.append(X_train_valid[closest_idx])
            if len(centroids_list) == 0:
                centroids_list = [X_train[0]]
            background = np.array(centroids_list)
            print(f"      ✓ Frequency clustering: {len(background)} representative samples")
            
        elif bg_type == 'wavelet_clustering':
            n_clusters = int(bg_params.get('n_clusters', 10))
            n_clusters = min(n_clusters, len(X_train))
            print(f"      Computing wavelet features for {len(X_train)} samples...")
            features = []
            for i, sample in enumerate(X_train):
                if i % 500 == 0 and i > 0:
                    print(f"        Progress: {i}/{len(X_train)}")
                try:
                    wav_feat = compute_wavelet_features(sample)
                    features.append(wav_feat)
                except Exception as e:
                    print(f"        Warning: Sample {i} failed wavelet extraction: {e}")
                    features.append(np.zeros(20))
            features = np.array(features)
            print(f"      Features shape: {features.shape}")
            valid_mask = np.all(np.isfinite(features), axis=1)
            if not np.all(valid_mask):
                print(f"      Warning: {np.sum(~valid_mask)} samples have invalid features")
                features = features[valid_mask]
                X_train_valid = X_train[valid_mask]
            else:
                X_train_valid = X_train
            if len(features) < n_clusters:
                n_clusters = len(features)
            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
            labels = kmeans.fit_predict(features)
            centroids_list = []
            for k in range(n_clusters):
                cluster_indices = np.where(labels == k)[0]
                if len(cluster_indices) > 0:
                    cluster_features = features[cluster_indices]
                    centroid = kmeans.cluster_centers_[k]
                    distances = np.linalg.norm(cluster_features - centroid, axis=1)
                    closest_idx = cluster_indices[np.argmin(distances)]
                    centroids_list.append(X_train_valid[closest_idx])
            if len(centroids_list) == 0:
                centroids_list = [X_train[0]]
            background = np.array(centroids_list)
            print(f"      ✓ Wavelet clustering: {len(background)} representative samples")
            
        elif bg_type == 'matrix_profile_sampling':
            n_samples = int(bg_params.get('n_samples', 10))
            window_size = int(bg_params.get('window_size', 10))
            if not STUMPY_AVAILABLE:
                print(f"      Warning: STUMPY not available, falling back to random sampling")
                indices = np.random.choice(len(X_train), min(n_samples, len(X_train)), replace=False)
                background = X_train[indices]
            else:
                max_analyze = min(100, len(X_train))
                print(f"      Computing matrix profiles for {max_analyze} samples...")
                mp_scores = []
                for i in range(max_analyze):
                    mp = compute_matrix_profile(X_train[i], window_size)
                    if mp is not None:
                        mp_scores.append((i, np.mean(mp)))
                if len(mp_scores) > 0:
                    mp_scores.sort(key=lambda x: x[1])
                    n_select = min(n_samples, len(mp_scores))
                    selected_indices = [idx for idx, _ in mp_scores[:n_select]]
                    background = X_train[selected_indices]
                else:
                    indices = np.random.choice(len(X_train), min(n_samples, len(X_train)), replace=False)
                    background = X_train[indices]
            print(f"      ✓ Matrix profile sampling: {len(background)} samples")
            
        elif bg_type == 'feature_mean':
            feature_means = np.mean(X_train, axis=0, keepdims=True)
            background = feature_means
            print(f"      ✓ Feature mean: {len(background)} sample")
            
        else:
            raise ValueError(f"Unknown background type: {bg_type}")
        
        return background.astype(np.float32)
    
    except Exception as e:
        print(f"      ❌ Background generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def build_optimized_background_configs(primary_use, option_number, X_train):
    """Build background configurations with optimized cluster numbers"""
    print(f"\n  📊 Building optimized background configurations...")
    
    configs = []
    
    # Add random sampling if enabled
    if ENABLE_RANDOM_SAMPLE:
        for n_samples in RANDOM_SAMPLE_COUNTS:
            configs.append({'type': 'random_sample', 'params': {'n_samples': n_samples}})
        print(f"      ✅ Random sampling: ENABLED ({len(RANDOM_SAMPLE_COUNTS)} configs)")
    else:
        print(f"      ⏭️  Random sampling: SKIPPED")
    
    # Add matrix profile if enabled
    if ENABLE_MATRIX_PROFILE:
        mp_count = 0
        for n_samples in MATRIX_PROFILE_SAMPLE_COUNTS:
            for window_size in MATRIX_PROFILE_WINDOW_SIZES:
                configs.append({
                    'type': 'matrix_profile_sampling',
                    'params': {'n_samples': n_samples, 'window_size': window_size}
                })
                mp_count += 1
        print(f"      ✅ Matrix profile: ENABLED ({mp_count} configs)")
    else:
        print(f"      ⏭️  Matrix profile: SKIPPED")
    
    # Add feature mean if enabled
    if ENABLE_FEATURE_MEAN:
        configs.append({'type': 'feature_mean', 'params': {}})
        print(f"      ✅ Feature mean: ENABLED")
    else:
        print(f"      ⏭️  Feature mean: SKIPPED")
    
    # Add clustering configs with optimized k values
    clustering_methods = []
    if ENABLE_KMEANS:
        clustering_methods.append('kmeans')
    if ENABLE_FREQUENCY_CLUSTERING:
        clustering_methods.append('frequency_clustering')
    if ENABLE_WAVELET_CLUSTERING:
        clustering_methods.append('wavelet_clustering')
    
    for method in clustering_methods:
        # Try to load from database first
        optimal_k = load_optimal_clusters(primary_use, option_number, method)
        
        # If not in database, compute optimal k
        if optimal_k is None and CLUSTER_OPTIMIZATION_ENABLED:
            print(f"\n      🔍 Running cluster optimization for {method}...")
            optimal_k = find_optimal_clusters(X_train, method, primary_use, option_number)
        
        # If optimization disabled or failed, use default values
        if optimal_k is None:
            print(f"      ⚠️  Using default clusters for {method}: [5, 20]")
            optimal_k = [5, 20]
        
        # Add configs for each recommended k
        for k in optimal_k:
            configs.append({
                'type': method,
                'params': {'n_clusters': k}
            })
        
        print(f"      ✅ {method}: k={optimal_k} ({len(optimal_k)} configs)")
    
    # Show which clustering methods were skipped
    if not ENABLE_KMEANS:
        print(f"      ⏭️  kmeans: SKIPPED")
    if not ENABLE_FREQUENCY_CLUSTERING:
        print(f"      ⏭️  frequency_clustering: SKIPPED")
    if not ENABLE_WAVELET_CLUSTERING:
        print(f"      ⏭️  wavelet_clustering: SKIPPED")
    
    print(f"\n  📦 Total background configurations: {len(configs)}")
    
    return configs


# ============================
# BACKGROUND DATABASE FUNCTIONS
# ============================

def save_background_to_table(primary_use, option_number, bg_type, bg_params, 
                             background, test_indices, generation_time, db_path=BACKGROUND_DB):
    """Save background data to database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Convert numpy types to native Python types for JSON serialization
    bg_params_clean = {}
    for key, value in bg_params.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            bg_params_clean[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            bg_params_clean[key] = float(value)
        else:
            bg_params_clean[key] = value
    
    bg_params_str = json.dumps(bg_params_clean, sort_keys=True)
    bg_json = json.dumps(background.tolist())
    test_indices_json = json.dumps([int(idx) for idx in test_indices])
    shape_info = json.dumps(list(background.shape))
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT OR REPLACE INTO background_data 
        (primary_use, option_number, background_type, background_params,
         background_data_json, test_indices_json, background_size, 
         generation_time, shape_info, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, bg_type, bg_params_str,
          bg_json, test_indices_json, len(background),
          generation_time, shape_info, timestamp))
    
    conn.commit()
    conn.close()


def load_background_from_table(primary_use, option_number, bg_type, bg_params, db_path=BACKGROUND_DB):
    """Load background data from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    bg_params_str = json.dumps(bg_params, sort_keys=True)
    
    cursor.execute('''
        SELECT background_data_json, test_indices_json, background_size, generation_time, timestamp
        FROM background_data
        WHERE primary_use = ? AND option_number = ? 
        AND background_type = ? AND background_params = ?
    ''', (primary_use, option_number, bg_type, bg_params_str))
    
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        return None
    
    bg_json, test_indices_json, bg_size, gen_time, timestamp = result
    background = np.array(json.loads(bg_json), dtype=np.float32)
    test_indices = json.loads(test_indices_json)
    
    return {
        'data': background,
        'test_indices': test_indices,
        'size': bg_size,
        'generation_time': gen_time,
        'timestamp': timestamp
    }


def check_existing_background(primary_use, option_number, bg_type, bg_params, db_path=BACKGROUND_DB):
    """Check if background already exists in database"""
    result = load_background_from_table(primary_use, option_number, bg_type, bg_params, db_path)
    return result is not None


# ============================
# DATASET LOADING
# ============================

def load_dataset(primary_use, option_number):
    """Load dataset using preprocess function"""
    from Functions import preprocess
    container = preprocess.load_and_preprocess_data_with_sequences(
        db_path="energy_data.db",
        primary_use=primary_use,
        option_number=option_number,
        scaled=True,
        scale_type="both"
    )
    return container


def get_all_datasets_from_benchmark(db_path=BENCHMARK_DB):
    """Get all unique datasets from benchmark database"""
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT DISTINCT primary_use, option_number 
        FROM prediction_performance
        ORDER BY primary_use, option_number
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ============================
# PROGRESS TRACKING
# ============================

def get_background_status(primary_use, option_number, db_path=BACKGROUND_DB):
    """Get background generation status for a dataset"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT COUNT(*) FROM background_data 
        WHERE primary_use = ? AND option_number = ?
    ''', (primary_use, option_number))
    
    count = cursor.fetchone()[0]
    conn.close()
    
    return count


def print_progress_table(datasets_df, db_path=BACKGROUND_DB):
    """Print progress table for all datasets"""
    print("\n" + "="*100)
    print("📊 BACKGROUND GENERATION PROGRESS")
    print("="*100)
    print(f"{'Primary Use':<20} {'Option':<10} {'Backgrounds':<15} {'Status':<20}")
    print("-"*100)
    
    total_complete = 0
    total_datasets = len(datasets_df)
    
    for _, row in datasets_df.iterrows():
        primary_use = row['primary_use']
        option_number = int(row['option_number'])
        
        bg_count = get_background_status(primary_use, option_number, db_path)
        
        if bg_count > 0:
            status = f"✅ {bg_count} generated"
            total_complete += 1
        else:
            status = "⚠️  Pending"
        
        print(f"{primary_use:<20} {option_number:<10} {bg_count:<15} {status:<20}")
    
    print("-"*100)
    completion_pct = (total_complete / total_datasets * 100) if total_datasets > 0 else 0
    print(f"TOTAL: {total_complete}/{total_datasets} datasets complete ({completion_pct:.1f}%)")
    print("="*100)


# ============================
# MAIN GENERATION FUNCTION
# ============================

def generate_all_backgrounds(replace_existing=False):
    """Generate backgrounds for all datasets from benchmark database"""
    print("\n" + "="*80)
    print("🔧 BACKGROUND DATA GENERATION")
    print("="*80)
    
    # Get all datasets from benchmark
    datasets_df = get_all_datasets_from_benchmark()
    print(f"\nFound {len(datasets_df)} datasets in benchmark database")
    
    # Show initial progress
    print_progress_table(datasets_df)
    
    # Process each dataset
    for idx, row in datasets_df.iterrows():
        primary_use = row['primary_use']
        option_number = int(row['option_number'])
        
        print(f"\n{'='*80}")
        print(f"📦 DATASET {idx + 1}/{len(datasets_df)}: {primary_use} - Option {option_number}")
        print(f"{'='*80}")
        
        try:
            # Load dataset
            print(f"  Loading dataset...")
            container = load_dataset(primary_use, option_number)
            X_train = container.X_train
            X_test = container.X_test
            n_test_samples = X_test.shape[0]
            
            print(f"  Dataset shape: Train={X_train.shape}, Test={X_test.shape}")
            
            # Get or generate test indices
            test_indices = get_or_generate_test_indices(primary_use, option_number, n_test_samples)
            
            # Build background configurations
            bg_configs = build_optimized_background_configs(primary_use, option_number, X_train)
            
            # Process each background configuration
            for bg_idx, bg_config in enumerate(bg_configs):
                bg_type = bg_config['type']
                bg_params = bg_config['params']
                
                # Convert numpy types to native Python types
                bg_params_clean = {}
                for key, value in bg_params.items():
                    if isinstance(value, (np.integer, np.int64, np.int32)):
                        bg_params_clean[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        bg_params_clean[key] = float(value)
                    else:
                        bg_params_clean[key] = value
                
                bg_params_str = json.dumps(bg_params_clean, sort_keys=True)
                
                # Check if already exists
                if check_existing_background(primary_use, option_number, bg_type, bg_params_clean):
                    if not replace_existing:
                        print(f"    [{bg_idx+1}/{len(bg_configs)}] {bg_type} {bg_params_str} - Already exists, SKIPPING")
                        continue
                    else:
                        print(f"    [{bg_idx+1}/{len(bg_configs)}] {bg_type} {bg_params_str} - Replacing existing...")
                else:
                    print(f"    [{bg_idx+1}/{len(bg_configs)}] Generating {bg_type} {bg_params_str}...")
                
                try:
                    start_time = time.time()
                    background = generate_background_data(X_train, bg_type, bg_params_clean)
                    gen_time = time.time() - start_time
                    
                    save_background_to_table(
                        primary_use, option_number, bg_type, bg_params_clean,
                        background, test_indices, gen_time
                    )
                    
                    print(f"        ✅ Saved to database: {len(background)} samples, {gen_time:.2f}s")
                
                except Exception as e:
                    print(f"        ❌ Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n  ✅ Completed {primary_use} - Option {option_number}")
            
        except Exception as e:
            print(f"\n  ❌ Failed to process dataset: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("✅ BACKGROUND GENERATION COMPLETE")
    print("="*80)
    
    # Show final progress
    print_progress_table(datasets_df)


# ============================
# UTILITY/REPORTING FUNCTIONS
# ============================

def print_database_summary():
    """Print summary of background database"""
    conn = sqlite3.connect(BACKGROUND_DB)
    
    print("\n" + "="*80)
    print("📊 BACKGROUND DATABASE SUMMARY")
    print("="*80)
    
    # Background data count
    query_bg = 'SELECT COUNT(*) FROM background_data'
    cursor = conn.cursor()
    cursor.execute(query_bg)
    n_backgrounds = cursor.fetchone()[0]
    print(f"\n  Total backgrounds: {n_backgrounds}")
    
    # By background type
    query_type = '''
        SELECT background_type, COUNT(*) 
        FROM background_data 
        GROUP BY background_type
    '''
    df_type = pd.read_sql_query(query_type, conn)
    print("\n  By background type:")
    for _, row in df_type.iterrows():
        print(f"    • {row['background_type']}: {row['COUNT(*)']}")
    
    # By dataset
    query_dataset = '''
        SELECT primary_use, COUNT(*) 
        FROM background_data 
        GROUP BY primary_use
    '''
    df_dataset = pd.read_sql_query(query_dataset, conn)
    print("\n  By dataset:")
    for _, row in df_dataset.iterrows():
        print(f"    • {row['primary_use']}: {row['COUNT(*)']}")
    
    # Cluster optimizations
    query_opt = 'SELECT COUNT(DISTINCT primary_use || option_number || clustering_type) FROM cluster_optimization'
    cursor.execute(query_opt)
    n_optimizations = cursor.fetchone()[0]
    print(f"\n  Cluster optimizations: {n_optimizations}")
    
    # Test indices
    query_indices = 'SELECT COUNT(*) FROM test_sample_indices'
    cursor.execute(query_indices)
    n_indices = cursor.fetchone()[0]
    print(f"  Test sample indices: {n_indices}")
    
    conn.close()
    print("\n" + "="*80)


def print_configuration():
    """Print current configuration settings"""
    print("\n" + "="*80)
    print("⚙️  CURRENT CONFIGURATION SETTINGS")
    print("="*80)
    
    print("\n🎯 ENABLED BACKGROUND TYPES:")
    print(f"  • Random Sample:         {ENABLE_RANDOM_SAMPLE}")
    if ENABLE_RANDOM_SAMPLE:
        print(f"    Sample counts: {RANDOM_SAMPLE_COUNTS}")
    
    print(f"  • K-Means:               {ENABLE_KMEANS}")
    print(f"  • Frequency Clustering:  {ENABLE_FREQUENCY_CLUSTERING}")
    print(f"  • Wavelet Clustering:    {ENABLE_WAVELET_CLUSTERING}")
    
    print(f"  • Matrix Profile:        {ENABLE_MATRIX_PROFILE}")
    if ENABLE_MATRIX_PROFILE:
        mp_total = len(MATRIX_PROFILE_SAMPLE_COUNTS) * len(MATRIX_PROFILE_WINDOW_SIZES)
        print(f"    Sample counts:       {MATRIX_PROFILE_SAMPLE_COUNTS}")
        print(f"    Window sizes:        {MATRIX_PROFILE_WINDOW_SIZES}")
        print(f"    Total combinations:  {mp_total}")
    
    print(f"  • Feature Mean:          {ENABLE_FEATURE_MEAN}")
    
    print("\n🔧 CLUSTER OPTIMIZATION:")
    print(f"  • Enabled:               {CLUSTER_OPTIMIZATION_ENABLED}")
    print(f"  • K range:               {min(CLUSTER_RANGE)} to {max(CLUSTER_RANGE)}")
    print(f"  • Sample percentage:     {CLUSTER_SAMPLE_PCT*100:.0f}%")
    
    print("\n📊 GENERAL SETTINGS:")
    print(f"  • Test samples per cfg:  {N_SAMPLES_PER_CONFIG}")
    print(f"  • Random seed:           {RANDOM_SEED}")
    
    print("\n" + "="*80)


def view_test_indices():
    """View all test indices"""
    print("\n" + "="*80)
    print("📌 TEST SAMPLE INDICES")
    print("="*80)
    
    conn = sqlite3.connect(BACKGROUND_DB)
    query = '''
        SELECT primary_use, option_number, test_indices_json, 
               n_samples, random_seed, timestamp
        FROM test_sample_indices
        ORDER BY primary_use, option_number
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        print("\nNo test indices found.")
    else:
        print(f"\nStored test indices for {len(df)} dataset(s):")
        for _, row in df.iterrows():
            indices = json.loads(row['test_indices_json'])
            print(f"\n  {row['primary_use']} - Option {row['option_number']}:")
            print(f"    Generated: {row['timestamp'][:10]}")
            print(f"    Random seed: {row['random_seed']}")
            print(f"    N samples: {row['n_samples']}")
            print(f"    Indices: {indices}")
    
    print("\n" + "="*80)


def view_cluster_results():
    """View cluster optimization results"""
    print("\n" + "="*80)
    print("🔍 CLUSTER OPTIMIZATION RESULTS")
    print("="*80)
    
    conn = sqlite3.connect(BACKGROUND_DB)
    query = '''
        SELECT primary_use, option_number, clustering_type, k,
               silhouette_score, davies_bouldin_index, 
               calinski_harabasz_score, dunn_index, is_recommended
        FROM cluster_optimization
        WHERE is_recommended = 1
        ORDER BY primary_use, option_number, clustering_type, k
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        print("\nNo cluster optimization results found.")
    else:
        print(f"\nRecommended cluster configurations: {len(df)}")
        for (primary_use, option_number, clustering_type), group in df.groupby(['primary_use', 'option_number', 'clustering_type']):
            recommended_k = group['k'].tolist()
            print(f"\n  {primary_use} - Option {option_number} - {clustering_type}:")
            print(f"    Recommended k: {recommended_k}")
            for _, row in group.iterrows():
                print(f"      k={int(row['k'])}: Silhouette={row['silhouette_score']:.3f}, "
                      f"DB={row['davies_bouldin_index']:.3f}, "
                      f"CH={row['calinski_harabasz_score']:.1f}, "
                      f"Dunn={row['dunn_index']:.3f}")
    
    print("\n" + "="*80)


# ============================
# MAIN FUNCTION
# ============================

def main():
    """Main entry point"""
    import sys
    
    # Initialize database
    init_background_database()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--help" or command == "-h":
            print("\nBackground Data Generation System")
            print("="*80)
            print("\nUSAGE:")
            print("  python background_gen.py              - Generate backgrounds for all datasets")
            print("  python background_gen.py --replace    - Regenerate all backgrounds (replace existing)")
            print("\nCOMMANDS:")
            print("  python background_gen.py config       - Show configuration settings")
            print("  python background_gen.py status       - Show generation progress")
            print("  python background_gen.py summary      - Show database summary")
            print("  python background_gen.py indices      - View test sample indices")
            print("  python background_gen.py clusters     - View cluster optimization results")
            print("\nFEATURES:")
            print("  ✅ Automatic processing of all datasets from benchmark_results.db")
            print("  ✅ Cluster optimization using 4 metrics (Silhouette, Davies-Bouldin, etc.)")
            print("  ✅ Permanent test indices (generated once, reused forever)")
            print("  ✅ Skip existing backgrounds or replace with --replace flag")
            print("  ✅ Configuration flags to enable/disable background types")
            
        elif command == "--replace":
            print("\n⚠️  REPLACE MODE: Will regenerate ALL backgrounds")
            confirm = input("Are you sure? This will overwrite existing data. (yes/no): ").strip().lower()
            if confirm == 'yes':
                generate_all_backgrounds(replace_existing=True)
            else:
                print("Cancelled.")
        
        elif command == "config":
            print_configuration()
        
        elif command == "status":
            datasets_df = get_all_datasets_from_benchmark()
            print_progress_table(datasets_df)
        
        elif command == "summary":
            print_database_summary()
        
        elif command == "indices":
            view_test_indices()
        
        elif command == "clusters":
            view_cluster_results()
        
        else:
            print(f"Unknown command: {command}")
            print("Run 'python background_gen.py --help' for usage")
    
    else:
        # Default: generate backgrounds (skip existing)
        print("\n⚠️  This will generate backgrounds for all datasets")
        print("   Existing backgrounds will be SKIPPED")
        print("   Use '--replace' flag to regenerate existing backgrounds")
        
        confirm = input("\nProceed? (yes/no): ").strip().lower()
        if confirm == 'yes':
            generate_all_backgrounds(replace_existing=False)
        else:
            print("Cancelled.")


if __name__ == "__main__":
    main()