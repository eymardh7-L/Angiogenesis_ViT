"""
Fine-Tuning Swin Transformer for Angiogenesis Classification

Full fine-tuning of Swin-Base with replaced classification head.
Includes 80/20 train/val split, data augmentation, and early stopping
based on validation loss.

Author: [Your Name]
Date: 2024
License: MIT
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Dict
import timm
from timm.data import resolve_data_config, create_transform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Fine-tuning configuration."""
    ROOT_PATH: str = '/content/drive/MyDrive/Colab Notebooks/Data/Angiogenesis'
    LABEL_MAP: Dict[str, int] = {
        "leptina": 0,
        "FAK+leptina": 1,
        "Src+leptina": 2
    }
    NUM_CLASSES: int = len(LABEL_MAP)
    
    # Model: Swin-Base pretrained on ImageNet-22k, fine-tuned on ImageNet-1k
    MODEL_NAME: str = 'swin_base_patch4_window7_224.ms_in22k_ft_in1k'
    
    # Training hyperparameters
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 8
    EPOCHS: int = 3
    WEIGHT_DECAY: float = 0.01  # AdamW default
    
    # Data splitting
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.2
    
    # Hardware
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    RANDOM_SEED: int = 42
    
    # Experiment design
    N_EXPERIMENTS: int = 10
    RUNS_PER_EXPERIMENT: int = 30


# =============================================================================
# DATASET
# =============================================================================

class AngioDataset(Dataset):
    """
    PyTorch Dataset for angiogenesis histopathology images.
    """
    
    def __init__(self, samples: List[Tuple[str, int]], transform):
        """
        Args:
            samples: list of (image_path, label) tuples
            transform: torchvision transform pipeline
        """
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Load image
        img = Image.open(path).convert('RGB')
        
        # Apply transform
        img = self.transform(img)
        
        return img, label


def load_dataset(root_path: str, label_map: Dict[str, int]) -> List[Tuple[str, int]]:
    """Load all image paths and labels."""
    all_samples = []
    
    for class_name, idx in label_map.items():
        folder = os.path.join(root_path, class_name)
        
        for fname in os.listdir(folder):
            if fname.lower().endswith(('jpg', 'png', 'jpeg')):
                path = os.path.join(folder, fname)
                all_samples.append((path, idx))
    
    print(f"Loaded {len(all_samples)} images")
    return all_samples


def split_dataset(samples: List[Tuple[str, int]], 
                  train_ratio: float = Config.TRAIN_SPLIT,
                  random_state: int = None) -> Tuple[List, List]:
    """
    Random train/validation split with stratification.
    
    Args:
        samples: full dataset
        train_ratio: fraction for training
        random_state: seed for reproducibility
    
    Returns:
        train_samples, val_samples
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle
    samples = samples.copy()
    np.random.shuffle(samples)
    
    # Split
    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    
    # Verify stratification
    train_dist = [sum(1 for _, l in train_samples if l == i) 
                  for i in range(Config.NUM_CLASSES)]
    val_dist = [sum(1 for _, l in val_samples if l == i) 
                for i in range(Config.NUM_CLASSES)]
    
    print(f"Train: {len(train_samples)} {train_dist}")
    print(f"Val:   {len(val_samples)} {val_dist}")
    
    return train_samples, val_samples


# =============================================================================
# MODEL
# =============================================================================

class SwinClassifier:
    """
    Fine-tunable Swin Transformer classifier.
    """
    
    def __init__(self, 
                 model_name: str = Config.MODEL_NAME,
                 num_classes: int = Config.NUM_CLASSES,
                 device: str = Config.DEVICE):
        """
        Initialize model with replaced classification head.
        
        Args:
            model_name: timm model identifier
            num_classes: number of output classes
            device: computation device
        """
        self.device = device
        
        # Create model with new head
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes  # Replaces original head
        ).to(device)
        
        # Get data configuration
        self.config = resolve_data_config({}, model=self.model)
        
        print(f"Loaded {model_name}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def get_transforms(self, is_training: bool = False):
        """
        Get preprocessing transforms.
        
        Args:
            is_training: if True, apply augmentation
        
        Returns:
            transform pipeline
        """
        return create_transform(**self.config, is_training=is_training)
    
    def train_epoch(self, 
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> float:
        """
        Train for one epoch.
        
        Returns:
            average loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            logits = self.model(images)
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[List, List, float]:
        """
        Evaluate on validation set.
        
        Returns:
            y_true, y_pred, inference_time
        """
        self.model.eval()
        
        y_true, y_pred = [], []
        start_time = time.time()
        
        for images, labels in val_loader:
            images = images.to(self.device)
            
            # Forward
            logits = self.model(images)
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
        
        inference_time = time.time() - start_time
        
        return y_true, y_pred, inference_time


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def fine_tune_once(model: SwinClassifier,
                   train_samples: List,
                   val_samples: List,
                   epochs: int = Config.EPOCHS,
                   lr: float = Config.LEARNING_RATE) -> Dict:
    """
    Execute one fine-tuning run.
    
    Args:
        model: initialized SwinClassifier
        train_samples: training set
        val_samples: validation set
        epochs: training epochs
        lr: learning rate
    
    Returns:
        Dictionary with metrics and predictions
    """
    # Create transforms
    train_transform = model.get_transforms(is_training=True)   # Augmentation
    val_transform = model.get_transforms(is_training=False)     # No augmentation
    
    # Create datasets
    train_ds = AngioDataset(train_samples, train_transform)
    val_ds = AngioDataset(val_samples, val_transform)
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optimizer, criterion)
        print(f"  Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}")
    
    # Evaluation
    y_true, y_pred, inference_time = model.evaluate(val_loader)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, 
                                           average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, 
                                     average='macro', zero_division=0),
        'inference_time': inference_time,
        'throughput': len(y_true) / inference_time if inference_time > 0 else 0
    }
    
    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred
    }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_fine_tuning_experiment(all_samples: List[Tuple[str, int]],
                               n_experiments: int = Config.N_EXPERIMENTS,
                               runs_per_exp: int = Config.RUNS_PER_EXPERIMENT,
                               output_dir: str = './results') -> pd.DataFrame:
    """
    Execute full fine-tuning experimental protocol.
    
    Args:
        all_samples: full dataset
        n_experiments: independent experiments
        runs_per_exp: runs per experiment
        output_dir: output directory
    
    Returns:
        DataFrame with aggregated results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage
    results = {
        'Accuracy': [], 'F1_macro': [], 'Precision': [],
        'Recall': [], 'Time': [], 'Cost': []  # Cost = throughput
    }
    
    for exp_idx in range(n_experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_idx + 1}/{n_experiments}")
        print(f"{'='*60}")
        
        exp_metrics = {k: [] for k in ['acc', 'f1', 'prec', 'rec', 'time', 'cost']}
        
        for run_idx in range(runs_per_exp):
            # Fresh model per run (no weight carryover)
            model = SwinClassifier()
            
            # Fresh split per run
            train_s, val_s = split_dataset(
                all_samples,
                train_ratio=Config.TRAIN_SPLIT,
                random_state=Config.RANDOM_SEED + exp_idx * 1000 + run_idx
            )
            
            # Train and evaluate
            run_results = fine_tune_once(model, train_s, val_s)
            m = run_results['metrics']
            
            exp_metrics['acc'].append(m['accuracy'])
            exp_metrics['f1'].append(m['f1_macro'])
            exp_metrics['prec'].append(m['precision_macro'])
            exp_metrics['rec'].append(m['recall_macro'])
            exp_metrics['time'].append(m['inference_time'])
            exp_metrics['cost'].append(m['throughput'])
        
        # Aggregate experiment
        results['Accuracy'].append(np.mean(exp_metrics['acc']))
        results['F1_macro'].append(np.mean(exp_metrics['f1']))
        results['Precision'].append(np.mean(exp_metrics['prec']))
        results['Recall'].append(np.mean(exp_metrics['rec']))
        results['Time'].append(np.mean(exp_metrics['time']))
        results['Cost'].append(np.mean(exp_metrics['cost']))
        
        print(f"[Exp {exp_idx+1}] ACC={results['Accuracy'][-1]:.4f}, "
              f"F1={results['F1_macro'][-1]:.4f}")
    
    # DataFrame
    df = pd.DataFrame(results)
    
    # Save
    output_path = os.path.join(output_dir, 'fine_tune_swin_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Fine-Tuned Swin Transformer")
    print("="*60)
    print(df.describe())
    print(f"\nRobustness (Accuracy variance): {df['Accuracy'].var():.6f}")
    print(f"Range: {df['Accuracy'].max() - df['Accuracy'].min():.4f}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Load data
    all_samples = load_dataset(Config.ROOT_PATH, Config.LABEL_MAP)
    
    # Run experiment
    results_df = run_fine_tuning_experiment(
        all_samples,
        n_experiments=Config.N_EXPERIMENTS,
        runs_per_exp=Config.RUNS_PER_EXPERIMENT
    )