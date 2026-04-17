"""
Few-Shot Angiogenesis Classification with Proto-ConvNeXt

This module implements k-shot prototype learning for histopathological 
angiogenesis classification. Uses ConvNeXt-base for feature extraction and 
k-means clustering to compute class prototypes. Classification is performed 
by nearest-prototype assignment.


    Compute class prototypes via k-means clustering.
    
    Mathematical formulation:
        Given support set S_c = {x_{c,1}, ..., x_{c,k}} for class c,
        extract features: f_{c,i} = E(x_{c,i}) / ||E(x_{c,i})||_2
        
        Compute prototypes via k-means (n=2):
        {p_{c,1}, p_{c,2}} = k-means({f_{c,1}, ..., f_{c,k}})
    
    Classification: ŷ = argmin_{c,j} ||f_query - p_{c,j}||_2



Date: April/2026
Under: MIT
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import timm
from timm.data import resolve_data_config, create_transform


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Experiment configuration."""
    ROOT_PATH: str = '/content/drive/MyDrive/Colab Notebooks/Data/Angiogenesis'
    LABEL_MAP: Dict[str, int] = {
        "leptina": 0,           # Leptin control
        "FAK+leptina": 1,      # FAK inhibition
        "Src+leptina": 2        # Src inhibition
    }
    NUM_CLASSES: int = len(LABEL_MAP)
    
    # Model specification
    BACKBONE: str = 'convnext_base'  # ConvNeXt-base, 1024-dim features
    
    # Few-shot parameters
    K_SHOT: int = 5              # Support examples per class
    N_PROTOTYPES: int = 2        # k-means clusters per class
    
    # Hardware
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    RANDOM_SEED: int = 42
    
    # Experiment design
    N_EXPERIMENTS: int = 10      # Independent experiments
    RUNS_PER_EXPERIMENT: int = 30  # Runs per experiment (for robustness)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(root_path: str, 
                 label_map: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Load image dataset from directory structure.
    
    Returns:
        List of (image_path, label) tuples
    """
    dataset = []
    
    for class_name, label in label_map.items():
        folder = os.path.join(root_path, class_name)
        
        if not os.path.isdir(folder):
            print(f"Warning: {folder} not found")
            continue
        
        for fname in os.listdir(folder):
            if fname.lower().endswith(('jpg', 'png', 'jpeg')):
                path = os.path.join(folder, fname)
                dataset.append((path, label))
    
    print(f"Loaded {len(dataset)} images: "
          f"{[sum(1 for _, l in dataset if l == i) for i in range(len(label_map))]} "
          f"per class")
    return dataset


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class ConvNeXtFeatureExtractor:
    """
    ConvNeXt-based feature extractor with L2 normalization.
    """
    
    def __init__(self, model_name: str = Config.BACKBONE, 
                 device: str = Config.DEVICE):
        """
        Initialize ConvNeXt model without classification head.
        
        Args:
            model_name: timm model identifier
            device: computation device
        """
        self.device = device
        
        # Load model with num_classes=0 to remove head
        self.model = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0  # Feature extraction only
        ).to(device)
        
        self.model.eval()
        
        # Get preprocessing transforms from timm
        cfg = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**cfg, is_training=False)
        
        print(f"Loaded {model_name} on {device}")
        print(f"Output dimension: {self.model.num_features}")
    
    @torch.no_grad()
    def extract(self, image_path: str) -> np.ndarray:
        """
        Extract normalized feature vector from image.
        
        Args:
            image_path: path to image file
        
        Returns:
            L2-normalized feature vector (1024-dim for ConvNeXt-base)
        """
        # Load and preprocess
        img = Image.open(image_path).convert('RGB')
        x = self.transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        features = self.model(x)  # (1, 1024)
        features = features.squeeze(0).cpu().numpy()
        
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features


# =============================================================================
# PROTOTYPE CLASSIFICATION
# =============================================================================

class PrototypeClassifier:
    """
    k-shot prototype classifier using k-means clustering.
    """
    
    def __init__(self, 
                 k_shot: int = Config.K_SHOT,
                 n_prototypes: int = Config.N_PROTOTYPES,
                 feature_extractor: ConvNeXtFeatureExtractor = None):
        """
        Args:
            k_shot: number of support examples per class
            n_prototypes: number of k-means clusters per class
            feature_extractor: pre-initialized feature extractor
        """
        self.k_shot = k_shot
        self.n_prototypes = n_prototypes
        self.extractor = feature_extractor or ConvNeXtFeatureExtractor()
        self.prototypes = {}  # class_idx -> array of prototypes
    
    def build_support_set(self, 
                          dataset: List[Tuple[str, int]],
                          random_state: int = None) -> Tuple[Dict[int, List[str]], 
                                                              List[Tuple[str, int]]]:
        """
        Split dataset into support (k-shot) and query sets.
        
        Args:
            dataset: full dataset
            random_state: seed for reproducibility
        
        Returns:
            support: dict mapping class -> list of image paths
            query: list of (path, label) tuples for evaluation
        """
        if random_state:
            random.seed(random_state)
        
        random.shuffle(dataset)
        
        # Initialize support set for each class
        support = {label: [] for label in Config.LABEL_MAP.values()}
        query = []
        counts = {label: 0 for label in Config.LABEL_MAP.values()}
        
        # Allocate to support or query
        for path, label in dataset:
            if counts[label] < self.k_shot:
                support[label].append(path)
                counts[label] += 1
            else:
                query.append((path, label))
        
        # Verify support set size
        for label, paths in support.items():
            assert len(paths) == self.k_shot, \
                f"Class {label}: expected {self.k_shot} support, got {len(paths)}"
        
        print(f"Support: {self.k_shot} per class × {len(support)} classes = "
              f"{self.k_shot * len(support)} total")
        print(f"Query: {len(query)} images")
        
        return support, query
    
    def compute_prototypes(self, support: Dict[int, List[str]]) -> Dict[int, np.ndarray]:
        """
        Compute class prototypes via k-means clustering on support embeddings.
        
        Args:
            support: dict mapping class -> list of image paths
        
        Returns:
            prototypes: dict mapping class -> array of prototype vectors
        """
        prototypes = {}
        
        for class_idx, image_paths in support.items():
            # Extract features for all support images
            embeddings = [self.extractor.extract(p) for p in image_paths]
            embeddings = np.array(embeddings)
            
            # k-means clustering
            if len(embeddings) >= self.n_prototypes:
                kmeans = KMeans(
                    n_clusters=self.n_prototypes,
                    n_init=10,
                    random_state=Config.RANDOM_SEED
                )
                kmeans.fit(embeddings)
                prototypes[class_idx] = kmeans.cluster_centers_
            else:
                # Fallback: use raw embeddings if too few samples
                prototypes[class_idx] = embeddings
            
            print(f"Class {class_idx}: {len(embeddings)} support → "
                  f"{len(prototypes[class_idx])} prototypes")
        
        self.prototypes = prototypes
        return prototypes
    
    def predict(self, image_path: str) -> int:
        """
        Predict class for query image by nearest prototype.
        
        Args:
            image_path: path to query image
        
        Returns:
            predicted class index
        """
        # Extract query embedding
        query_emb = self.extractor.extract(image_path)
        
        # Find nearest prototype (any class)
        best_class = None
        best_distance = float('inf')
        
        for class_idx, proto_array in self.prototypes.items():
            for proto in proto_array:
                distance = np.linalg.norm(query_emb - proto)
                if distance < best_distance:
                    best_distance = distance
                    best_class = class_idx
        
        return best_class if best_class is not None else 0
    
    def evaluate(self, query_set: List[Tuple[str, int]]) -> Dict:
        """
        Evaluate on query set.
        
        Returns:
            Dictionary with predictions, labels, and metrics
        """
        y_true = []
        y_pred = []
        inference_times = []
        
        for img_path, true_label in query_set:
            start = time.time()
            pred_label = self.predict(img_path)
            elapsed = time.time() - start
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            inference_times.append(elapsed)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, 
                                               average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, 
                                         average='macro', zero_division=0),
            'mean_time': np.mean(inference_times),
            'std_time': np.std(inference_times),
            'throughput': len(y_true) / sum(inference_times) if sum(inference_times) > 0 else 0
        }
        
        return {
            'y_true': np.array(y_true),
            'y_pred': np.array(y_pred),
            'metrics': metrics
        }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_few_shot_experiment(dataset: List[Tuple[str, int]],
                            n_experiments: int = Config.N_EXPERIMENTS,
                            runs_per_exp: int = Config.RUNS_PER_EXPERIMENT,
                            output_dir: str = './results') -> pd.DataFrame:
    """
    Execute full few-shot experimental protocol.
    
    Design: n_experiments independent runs, each with runs_per_exp sub-runs
    for robustness estimation.
    
    Args:
        dataset: full dataset
        n_experiments: number of independent experiments
        runs_per_exp: runs per experiment
        output_dir: output directory
    
    Returns:
        DataFrame with aggregated results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize feature extractor (shared across all runs)
    extractor = ConvNeXtFeatureExtractor()
    
    # Storage for results
    results = {
        'Accuracy': [], 'F1_macro': [], 'Precision': [], 
        'Recall': [], 'Time': [], 'Throughput': []
    }
    
    for exp_idx in range(n_experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_idx + 1}/{n_experiments}")
        print(f"{'='*60}")
        
        exp_accs, exp_f1s, exp_precs, exp_recs, exp_times, exp_thrups = [], [], [], [], [], []
        
        for run_idx in range(runs_per_exp):
            # Initialize classifier with fresh random state per run
            classifier = PrototypeClassifier(
                k_shot=Config.K_SHOT,
                n_prototypes=Config.N_PROTOTYPES,
                feature_extractor=extractor  # Reuse extractor for efficiency
            )
            
            # Build support/query split
            support, query = classifier.build_support_set(
                dataset.copy(),
                random_state=Config.RANDOM_SEED + exp_idx * 1000 + run_idx
            )
            
            # Compute prototypes
            classifier.compute_prototypes(support)
            
            # Evaluate
            run_results = classifier.evaluate(query)
            m = run_results['metrics']
            
            exp_accs.append(m['accuracy'])
            exp_f1s.append(m['f1_macro'])
            exp_precs.append(m['precision_macro'])
            exp_recs.append(m['recall_macro'])
            exp_times.append(m['mean_time'])
            exp_thrups.append(m['throughput'])
        
        # Aggregate across runs in this experiment
        results['Accuracy'].append(np.mean(exp_accs))
        results['F1_macro'].append(np.mean(exp_f1s))
        results['Precision'].append(np.mean(exp_precs))
        results['Recall'].append(np.mean(exp_recs))
        results['Time'].append(np.mean(exp_times))
        results['Throughput'].append(np.mean(exp_thrups))
        
        print(f"[Exp {exp_idx+1}] ACC={results['Accuracy'][-1]:.4f}, "
              f"F1={results['F1_macro'][-1]:.4f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_path = os.path.join(output_dir, 'few_shot_proto_convnext_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Few-Shot Proto-ConvNeXt")
    print("="*60)
    print(df.describe())
    print(f"\nRobustness (Accuracy variance): {df['Accuracy'].var():.6f}")
    print(f"Range: {df['Accuracy'].max() - df['Accuracy'].min():.4f}")
    
    return df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_robustness_histogram(df: pd.DataFrame, 
                               metric: str = 'Accuracy',
                               output_path: str = None):
    """
    Plot distribution of metric across experiments for robustness visualization.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    plt.hist(df[metric], bins=10, edgecolor='black', alpha=0.7)
    plt.title(f'Robustness Assessment: {metric} Across {len(df)} Experiments')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.axvline(df[metric].mean(), color='red', linestyle='--', 
                label=f'Mean: {df[metric].mean():.4f}')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset(Config.ROOT_PATH, Config.LABEL_MAP)
    
    # Run full experiment
    results_df = run_few_shot_experiment(
        dataset,
        n_experiments=Config.N_EXPERIMENTS,
        runs_per_exp=Config.RUNS_PER_EXPERIMENT
    )
    
    # Optional: plot robustness
    plot_robustness_histogram(results_df, 'Accuracy', 
                              output_path='./results/few_shot_robustness.png')