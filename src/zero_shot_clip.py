"""
Zero-Shot Angiogenesis Classification with CLIP Ensemble Prompting

This module implements zero-shot classification of histopathological 
angiogenesis images using CLIP ViT-B/32 with ensemble prompting (100 prompts 
per class, 300 total). Prompt averaging enables robust prediction without 
task-specific fine-tuning.

Author: [Your Name]
Date: 2024
License: MIT
"""

import os
import time
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Tuple, Dict
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Experiment configuration parameters."""
    ROOT_PATH: str = '/content/drive/MyDrive/Colab Notebooks/Data/Angiogenesis'
    LABEL_MAP: Dict[str, int] = {
        "leptina": 0,           # Leptin-induced angiogenesis (control)
        "FAK+leptina": 1,      # FAK pathway inhibition
        "Src+leptina": 2        # Src pathway inhibition
    }
    CLASS_NAMES: List[str] = list(LABEL_MAP.keys())
    NUM_CLASSES: int = len(LABEL_MAP)
    
    # CLIP model specification
    CLIP_MODEL: str = "ViT-B/32"  # 512-dim image/text embeddings
    
    # Ensemble prompting parameters
    PROMPTS_PER_CLASS: int = 100
    TOTAL_PROMPTS: int = PROMPTS_PER_CLASS * NUM_CLASSES  # 300
    
    # Reproducibility
    RANDOM_SEED: int = 42
    
    # Hardware
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def generate_prompt_ensemble(class_names: List[str], 
                             prompts_per_class: int = 100) -> Tuple[List[str], Dict]:
    """
    Generate semantically varied prompt ensemble for zero-shot classification.
    
    Args:
        class_names: List of class identifiers (e.g., ["leptina", "FAK+leptina", "Src+leptina"])
        prompts_per_class: Number of prompt variations per class (default: 100)
    
    Returns:
        all_prompts: Flat list of all prompt strings (300 total)
        prompt_indices: Dictionary mapping class index to (start, end) indices in all_prompts
    
    Prompt templates:
        - Class 0: "Microscopy image showing leptin effect #{i}"
        - Class 1: "Microscopy image showing FAK and leptin interaction #{i}"
        - Class 2: "Microscopy image showing Src and leptin combined activity #{i}"
    
    Note:
        Numerical suffixes (#1-#100) enable statistical ensemble averaging 
        without semantic variation, isolating embedding stochasticity effects.
    """
    all_prompts = []
    prompt_indices = {}
    
    # Define semantic templates for each class
    templates = {
        "leptina": "Microscopy image showing leptin effect #{}",
        "FAK+leptina": "Microscopy image showing FAK and leptin interaction #{}",
        "Src+leptina": "Microscopy image showing Src and leptin combined activity #{}"
    }
    
    for class_idx, class_name in enumerate(class_names):
        start_idx = len(all_prompts)
        
        for i in range(1, prompts_per_class + 1):
            prompt = templates[class_name].format(i)
            all_prompts.append(prompt)
        
        end_idx = len(all_prompts)
        prompt_indices[class_idx] = (start_idx, end_idx)
    
    return all_prompts, prompt_indices


def save_prompts_to_json(prompts: List[str], 
                         indices: Dict, 
                         output_path: str = "prompt_templates.json"):
    """Save prompt ensemble to JSON for reproducibility and inspection."""
    data = {
        "prompts": prompts,
        "class_indices": indices,
        "metadata": {
            "total_prompts": len(prompts),
            "prompts_per_class": Config.PROMPTS_PER_CLASS,
            "num_classes": Config.NUM_CLASSES
        }
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Prompts saved to {output_path}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(root_path: str, 
                 label_map: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Load image paths and labels from directory structure.
    
    Expected directory structure:
        root_path/
            leptina/          -> class 0
                image1.jpg
                image2.png
            FAK+leptina/      -> class 1
                image3.jpg
            Src+leptina/      -> class 2
                image4.jpg
    
    Returns:
        List of (image_path, label) tuples
    """
    dataset = []
    
    for class_name, label in label_map.items():
        class_dir = os.path.join(root_path, class_name)
        
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                dataset.append((img_path, label))
    
    print(f"Loaded {len(dataset)} images from {len(label_map)} classes")
    return dataset


# =============================================================================
# INFERENCE
# =============================================================================

class CLIPZeroShotClassifier:
    """
    Zero-shot classifier using CLIP with ensemble prompt averaging.
    """
    
    def __init__(self, model_name: str = Config.CLIP_MODEL, device: str = Config.DEVICE):
        """
        Initialize CLIP model and preprocessing.
        
        Args:
            model_name: CLIP architecture (default: "ViT-B/32")
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()  # Inference mode
        
        print(f"Loaded CLIP {model_name} on {device}")
    
    def encode_text_ensemble(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode all prompt texts to normalized feature vectors.
        
        Args:
            prompts: List of prompt strings (300 total)
        
        Returns:
            text_features: (300, 512) tensor of L2-normalized text embeddings
        """
        # Tokenize all prompts
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def classify_image(self, 
                       image_path: str, 
                       text_features: torch.Tensor,
                       prompt_indices: Dict) -> Tuple[int, np.ndarray, float]:
        """
        Classify single image using ensemble prompt averaging.
        
        Args:
            image_path: Path to image file
            text_features: Pre-computed text embeddings (300, 512)
            prompt_indices: Dictionary mapping class -> (start, end) indices
        
        Returns:
            predicted_class: Integer class label (0, 1, or 2)
            class_probs: Probability distribution over classes
            inference_time: Wall-clock time for inference (seconds)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Time inference
        start_time = time.time()
        
        with torch.no_grad():
            # Encode image
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities with all prompts
            logits = image_features @ text_features.T  # (1, 300)
            probs = logits.softmax(dim=-1)  # (1, 300)
            
            # Average probabilities within each class ensemble
            class_probs = []
            for class_idx in sorted(prompt_indices.keys()):
                start, end = prompt_indices[class_idx]
                class_prob = probs[:, start:end].mean(dim=1).item()
                class_probs.append(class_prob)
            
            class_probs = np.array(class_probs)
            predicted_class = int(class_probs.argmax())
        
        inference_time = time.time() - start_time
        
        return predicted_class, class_probs, inference_time
    
    def evaluate_dataset(self,
                         dataset: List[Tuple[str, int]],
                         text_features: torch.Tensor,
                         prompt_indices: Dict) -> Dict:
        """
        Evaluate classifier on entire dataset.
        
        Args:
            dataset: List of (image_path, true_label) tuples
            text_features: Pre-computed text embeddings
            prompt_indices: Class -> prompt index mapping
        
        Returns:
            Dictionary with predictions, labels, inference times, and metrics
        """
        all_preds = []
        all_labels = []
        all_times = []
        
        for img_path, true_label in dataset:
            try:
                pred, probs, t_inf = self.classify_image(
                    img_path, text_features, prompt_indices
                )
                all_preds.append(pred)
                all_labels.append(true_label)
                all_times.append(t_inf)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'precision_macro': precision_score(all_labels, all_preds, 
                                               average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, 
                                         average='macro', zero_division=0),
            'mean_inference_time': np.mean(all_times),
            'std_inference_time': np.std(all_times),
            'throughput': len(all_labels) / np.sum(all_times) if sum(all_times) > 0 else 0
        }
        
        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'inference_times': np.array(all_times),
            'metrics': metrics
        }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(n_runs: int = 10,
                   save_results: bool = True,
                   output_dir: str = './results') -> np.ndarray:
    """
    Execute multiple independent experimental runs for robustness estimation.
    
    Args:
        n_runs: Number of independent runs (default: 10)
        save_results: Whether to save metrics to disk
        output_dir: Directory for output files
    
    Returns:
        metrics_matrix: (n_runs, 6) array of [accuracy, f1, prec, rec, time, throughput]
    """
    # Initialize
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(Config.RANDOM_SEED)
    
    # Load data
    dataset = load_dataset(Config.ROOT_PATH, Config.LABEL_MAP)
    
    # Generate prompts
    prompts, prompt_indices = generate_prompt_ensemble(
        Config.CLASS_NAMES, Config.PROMPTS_PER_CLASS
    )
    save_prompts_to_json(prompts, prompt_indices, 
                         os.path.join(output_dir, "prompt_templates.json"))
    
    # Initialize model
    classifier = CLIPZeroShotClassifier()
    text_features = classifier.encode_text_ensemble(prompts)
    
    # Run experiments
    metrics_matrix = []
    
    for run_idx in range(n_runs):
        print(f"\n{'='*50}")
        print(f"Run {run_idx + 1}/{n_runs}")
        print(f"{'='*50}")
        
        # Shuffle dataset for variability estimation
        shuffled_dataset = dataset.copy()
        np.random.shuffle(shuffled_dataset)
        
        # Evaluate
        results = classifier.evaluate_dataset(
            shuffled_dataset, text_features, prompt_indices
        )
        
        m = results['metrics']
        metrics_row = [
            m['accuracy'],
            m['f1_macro'],
            m['precision_macro'],
            m['recall_macro'],
            m['mean_inference_time'],
            m['throughput']
        ]
        metrics_matrix.append(metrics_row)
        
        print(f"Accuracy:  {m['accuracy']:.4f}")
        print(f"F1 Macro:  {m['f1_macro']:.4f}")
        print(f"Time:      {m['mean_inference_time']:.4f} ± {m['std_inference_time']:.4f} s")
    
    # Convert to array
    metrics_array = np.array(metrics_matrix)
    
    # Save results
    if save_results:
        output_path = os.path.join(output_dir, "zero_shot_metrics.npy")
        np.save(output_path, metrics_array)
        print(f"\nResults saved to {output_path}")
        
        # Also save as CSV for inspection
        import pandas as pd
        df = pd.DataFrame(metrics_array, 
                         columns=["Accuracy", "F1_macro", "Precision", 
                                  "Recall", "Avg_time", "Throughput"])
        df.to_csv(os.path.join(output_dir, "zero_shot_metrics.csv"), index=False)
        print(df.describe())
    
    return metrics_array


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run full experiment
    results = run_experiment(n_runs=10, save_results=True)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY: Zero-Shot CLIP Classification")
    print("="*50)
    print(f"Runs completed: {len(results)}")
    print(f"Mean accuracy:  {results[:, 0].mean():.4f} ± {results[:, 0].std():.4f}")
    print(f"Mean F1:        {results[:, 1].mean():.4f} ± {results[:, 1].std():.4f}")
    print(f"Mean time:      {results[:, 4].mean():.4f} ± {results[:, 4].std():.4f} s")