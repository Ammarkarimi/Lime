"""
Comprehensive evaluation metrics for LIME explanations.

Includes stability, consistency, faithfulness, and other metrics
to compare different explanation methods.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings


class ExplanationEvaluator:
    """Evaluator for comparing LIME explanations."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_explanation(
        self,
        explanation,
        X_neighborhood: np.ndarray,
        y_neighborhood: np.ndarray,
        label_idx: int,
        model_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single explanation.
        
        Args:
            explanation: LIME explanation object
            X_neighborhood: Neighborhood data used for explanation
            y_neighborhood: True predictions for neighborhood
            label_idx: Label index being explained
            model_predictions: Model predictions (if different from y_neighborhood)
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Model fit metrics
        if hasattr(explanation, 'score') and label_idx in explanation.score:
            metrics['r2_score'] = explanation.score[label_idx]
        else:
            # Compute R² manually
            if model_predictions is None:
                model_predictions = y_neighborhood
            # Get explanation predictions
            exp_predictions = self._get_explanation_predictions(
                explanation, X_neighborhood, label_idx
            )
            metrics['r2_score'] = r2_score(model_predictions, exp_predictions)
            metrics['mse'] = mean_squared_error(model_predictions, exp_predictions)
            metrics['mae'] = mean_absolute_error(model_predictions, exp_predictions)
        
        # Feature importance metrics
        feature_weights = dict(explanation.as_list(label=label_idx))
        metrics['num_features'] = len(feature_weights)
        metrics['max_importance'] = max(abs(w) for w in feature_weights.values()) if feature_weights else 0.0
        metrics['mean_importance'] = np.mean([abs(w) for w in feature_weights.values()]) if feature_weights else 0.0
        metrics['sparsity'] = sum(1 for w in feature_weights.values() if abs(w) < 0.01) / len(feature_weights) if feature_weights else 0.0
        
        return metrics
    
    def _get_explanation_predictions(
        self,
        explanation,
        X: np.ndarray,
        label_idx: int
    ) -> np.ndarray:
        """Get predictions from explanation model."""
        # Extract intercept and coefficients
        intercept = explanation.intercept[label_idx]
        feature_weights = dict(explanation.as_list(label=label_idx))
        
        # Get feature indices
        import re
        feature_indices = [int(feat.split('=')[0].split('_')[-1]) if '=' in feat else int(re.findall(r'\d+', feat)[0]) 
                          for feat in feature_weights.keys()]
        
        # Compute predictions
        predictions = np.full(len(X), intercept)
        for feat, weight in feature_weights.items():
            # Extract feature index (simplified - may need better parsing)
            try:
                if '=' in feat:
                    feat_idx = int(feat.split('=')[0].split('_')[-1])
                else:
                    import re
                    numbers = re.findall(r'\d+', feat)
                    feat_idx = int(numbers[0]) if numbers else 0
                
                if feat_idx < X.shape[1]:
                    predictions += weight * X[:, feat_idx]
            except:
                pass
        
        return predictions
    
    def evaluate_stability(
        self,
        explanations: List,
        label_idx: int
    ) -> Dict[str, float]:
        """
        Evaluate stability of explanations across multiple runs.
        
        Args:
            explanations: List of explanation objects (same instance, different runs)
            label_idx: Label index
        
        Returns:
            Stability metrics
        """
        if len(explanations) < 2:
            return {'stability_score': 1.0, 'mean_rank_correlation': 1.0}
        
        # Extract feature weights for each explanation
        all_weights = []
        all_features = set()
        
        for exp in explanations:
            weights = dict(exp.as_list(label=label_idx))
            all_weights.append(weights)
            all_features.update(weights.keys())
        
        # Create feature vectors
        feature_vectors = []
        for weights in all_weights:
            vec = np.array([weights.get(feat, 0.0) for feat in all_features])
            feature_vectors.append(vec)
        
        feature_vectors = np.array(feature_vectors)
        
        # Compute pairwise correlations
        from scipy.stats import spearmanr
        correlations = []
        for i in range(len(feature_vectors)):
            for j in range(i + 1, len(feature_vectors)):
                corr, _ = spearmanr(feature_vectors[i], feature_vectors[j])
                if not np.isnan(corr):
                    correlations.append(corr)
        
        stability_score = np.mean(correlations) if correlations else 0.0
        
        return {
            'stability_score': stability_score,
            'mean_rank_correlation': stability_score,
            'std_correlation': np.std(correlations) if correlations else 0.0
        }
    
    def compare_explanations(
        self,
        explanation1,
        explanation2,
        X_neighborhood: np.ndarray,
        y_neighborhood: np.ndarray,
        label_idx: int,
        model_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compare two explanations.
        
        Args:
            explanation1: First explanation
            explanation2: Second explanation
            X_neighborhood: Neighborhood data
            y_neighborhood: True predictions
            label_idx: Label index
            model_predictions: Model predictions
        
        Returns:
            Comparison metrics
        """
        # Evaluate each explanation
        metrics1 = self.evaluate_explanation(
            explanation1, X_neighborhood, y_neighborhood, label_idx, model_predictions
        )
        metrics2 = self.evaluate_explanation(
            explanation2, X_neighborhood, y_neighborhood, label_idx, model_predictions
        )
        
        # Compare feature importance
        weights1 = dict(explanation1.as_list(label=label_idx))
        weights2 = dict(explanation2.as_list(label=label_idx))
        
        all_features = set(weights1.keys()) | set(weights2.keys())
        
        # Compute rank correlation
        ranks1 = []
        ranks2 = []
        for feat in all_features:
            w1 = abs(weights1.get(feat, 0.0))
            w2 = abs(weights2.get(feat, 0.0))
            ranks1.append(w1)
            ranks2.append(w2)
        
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(ranks1, ranks2)
        
        # Compute feature agreement (top-k overlap)
        top_k = min(5, len(weights1), len(weights2))
        top_features1 = set(sorted(weights1.keys(), key=lambda x: abs(weights1[x]), reverse=True)[:top_k])
        top_features2 = set(sorted(weights2.keys(), key=lambda x: abs(weights2[x]), reverse=True)[:top_k])
        overlap = len(top_features1 & top_features2) / top_k if top_k > 0 else 0.0
        
        return {
            'r2_improvement': metrics2['r2_score'] - metrics1['r2_score'],
            'mse_improvement': metrics1.get('mse', 0) - metrics2.get('mse', 0),
            'rank_correlation': rank_corr if not np.isnan(rank_corr) else 0.0,
            'top_k_overlap': overlap,
            'explanation1_r2': metrics1['r2_score'],
            'explanation2_r2': metrics2['r2_score'],
        }


def compute_faithfulness(
    explanation,
    instance: np.ndarray,
    model_predict_fn,
    label_idx: int,
    num_samples: int = 100
) -> float:
    """
    Compute faithfulness metric: how well explanation predictions match model predictions.
    
    Args:
        explanation: LIME explanation
        instance: Instance being explained
        model_predict_fn: Model prediction function
        label_idx: Label index
        num_samples: Number of samples to test
    
    Returns:
        Faithfulness score (higher is better)
    """
    # Generate perturbed instances
    rng = np.random.RandomState(42)
    perturbations = []
    for _ in range(num_samples):
        perturbed = instance.copy()
        # Randomly set some features to zero (simulating feature removal)
        n_remove = rng.randint(1, len(instance))
        remove_indices = rng.choice(len(instance), n_remove, replace=False)
        perturbed[remove_indices] = 0
        perturbations.append(perturbed)
    
    perturbations = np.array(perturbations)
    
    # Get model predictions
    model_preds = model_predict_fn(perturbations)[:, label_idx]
    
    # Get explanation predictions (simplified - would need proper implementation)
    # For now, use a simplified version
    exp_preds = np.full(num_samples, explanation.intercept[label_idx])
    feature_weights = dict(explanation.as_list(label=label_idx))
    
    # This is a simplified version - proper implementation would need
    # to handle feature discretization and mapping
    for i, pert in enumerate(perturbations):
        for feat, weight in feature_weights.items():
            # Simplified: assume feature is present if value > 0
            try:
                feat_idx = int(feat.split('=')[0].split('_')[-1])
                if feat_idx < len(pert) and pert[feat_idx] > 0:
                    exp_preds[i] += weight
            except:
                pass
    
    # Compute correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(model_preds, exp_preds)
    
    return corr if not np.isnan(corr) else 0.0

