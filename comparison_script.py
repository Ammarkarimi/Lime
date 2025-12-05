"""
Comprehensive comparison script for different LIME explanation methods.

Compares:
1. Linear LIME (baseline)
2. LLM-enhanced LIME (OpenAI)
3. LLM-enhanced LIME (Anthropic)
4. Polynomial features LIME
5. Neural network LIME
6. Kernel Ridge LIME
"""

import os
import numpy as np
from typing import List, Dict, Tuple
import time
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

from lime.lime_tabular import LimeTabularExplainer
from lime.llm_lime_wrapper import LLMNonLinearRegressor, LLMEnhancedLimeExplainer
from lime.evaluation_metrics import ExplanationEvaluator, compute_faithfulness


def create_polynomial_regressor(max_degree=3, random_state=None):
    """Create polynomial features regressor."""
    poly = PolynomialFeatures(degree=max_degree, include_bias=False)
    ridge = Ridge(alpha=0.1, random_state=random_state)
    return Pipeline([('poly', poly), ('ridge', ridge)])


def create_neural_network_regressor(random_state=None):
    """Create neural network regressor."""
    return MLPRegressor(
        hidden_layer_sizes=(50, 25),
        max_iter=500,
        random_state=random_state,
        early_stopping=True
    )


def create_kernel_ridge_regressor():
    """Create kernel ridge regressor with RBF kernel."""
    return KernelRidge(alpha=0.1, kernel='rbf', gamma=0.1)


def compare_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    instance: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    num_features: int = 5,
    num_samples: int = 5000
) -> Dict[str, Dict]:
    """
    Compare different explanation methods.
    
    Returns:
        Dictionary with method names as keys and metrics as values
    """
    results = {}
    evaluator = ExplanationEvaluator()

    # Helper: adapt regressors so they accept sample_weight and expose coef_/intercept_
    from sklearn.linear_model import Ridge as _Ridge
    class _RegressorAdapter:
        def __init__(self, regressor):
            self._reg = regressor
            self.coef_ = None
            self.intercept_ = None

        def fit(self, X, y, sample_weight=None):
            # Try to pass sample_weight; if not supported, call without it.
            if sample_weight is not None:
                # If regressor is a Pipeline, forward sample_weight to the final step
                try:
                    if isinstance(self._reg, Pipeline):
                        last_step_name = self._reg.steps[-1][0]
                        params = {f"{last_step_name}__sample_weight": sample_weight}
                        self._reg.fit(X, y, **params)
                    else:
                        self._reg.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    # Underlying estimator doesn't accept sample_weight in fit
                    try:
                        # Last resort: try fitting without sample_weight
                        self._reg.fit(X, y)
                    except Exception:
                        raise
                except Exception:
                    # Any other exception: fallback to fit without sample_weight
                    self._reg.fit(X, y)
            else:
                self._reg.fit(X, y)

            # If underlying regressor exposes coef_, use it
            if hasattr(self._reg, 'coef_'):
                self.coef_ = getattr(self._reg, 'coef_')
            else:
                # Linearize the model locally: fit a Ridge to predict the regressor's outputs
                try:
                    preds = self._reg.predict(X)
                    ridge = _Ridge(alpha=1.0)
                    ridge.fit(X, preds)
                    self.coef_ = ridge.coef_
                    self.intercept_ = ridge.intercept_
                except Exception:
                    # Last resort: zeros
                    self.coef_ = np.zeros(X.shape[1])
                    self.intercept_ = 0.0

            if self.intercept_ is None:
                self.intercept_ = getattr(self._reg, 'intercept_', 0.0)

            return self

        def predict(self, X):
            return self._reg.predict(X)

        def score(self, X, y, sample_weight=None):
            if hasattr(self._reg, 'score'):
                try:
                    return self._reg.score(X, y, sample_weight=sample_weight)
                except TypeError:
                    return self._reg.score(X, y)
            return 0.0

    
    # Create base explainer
    base_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=0,
    )
    
    # Get prediction for this instance
    pred_proba = model.predict_proba(instance.reshape(1, -1))[0]
    pred_label_idx = int(np.argmax(pred_proba))
    
    print(f"\n{'='*80}")
    print(f"Comparing explanation methods for instance")
    print(f"Predicted class: {class_names[pred_label_idx]} (prob={pred_proba[pred_label_idx]:.4f})")
    print(f"{'='*80}\n")
    
    methods = {
        'Linear (Baseline)': {
            'regressor': None,  # Use default Ridge
            'provider': 'none'
        },
        'Polynomial Features': {
            'regressor': create_polynomial_regressor(max_degree=3, random_state=0),
            'provider': 'none'
        },
        'Neural Network': {
            'regressor': create_neural_network_regressor(random_state=0),
            'provider': 'none'
        },
        'Kernel Ridge (RBF)': {
            'regressor': create_kernel_ridge_regressor(),
            'provider': 'none'
        },
    }
    
    # Add LLM methods if API keys are available
    from lime.config import OPENAI_API_KEY, ANTHROPIC_API_KEY
    
    if OPENAI_API_KEY:
        methods['LLM (OpenAI)'] = {
            'regressor': None,
            'provider': 'openai',
            'use_llm': True
        }
    
    if ANTHROPIC_API_KEY:
        methods['LLM (Anthropic)'] = {
            'regressor': None,
            'provider': 'anthropic',
            'use_llm': True
        }
    
    # Evaluate each method
    for method_name, config in methods.items():
        print(f"\nEvaluating: {method_name}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            if config.get('use_llm', False):
                # Use LLM-enhanced explainer
                llm_explainer = LLMEnhancedLimeExplainer(
                    base_explainer=base_explainer,
                    use_llm_model=True,
                    importance_method='gradient',
                    llm_provider=config['provider'],
                    llm_model='gpt-4o-mini' if config['provider'] == 'openai' else None,
                    use_simple_polynomial=False,  # Don't fallback for comparison
                )
                
                explanation = llm_explainer.explain_instance(
                    data_row=instance,
                    predict_fn=model.predict_proba,
                    num_features=num_features,
                    num_samples=num_samples,
                )
            else:
                # Use custom regressor or default
                if config['regressor'] is not None:
                    explanation = base_explainer.explain_instance(
                        data_row=instance,
                        predict_fn=model.predict_proba,
                        num_features=num_features,
                        num_samples=num_samples,
                        model_regressor=config['regressor']
                    )
                else:
                    explanation = base_explainer.explain_instance(
                        data_row=instance,
                        predict_fn=model.predict_proba,
                        num_features=num_features,
                        num_samples=num_samples,
                    )
            
            elapsed_time = time.time() - start_time
            
            # Evaluate explanation
            # Note: We need to get neighborhood data for proper evaluation
            # For now, use the explanation's internal metrics
            metrics = evaluator.evaluate_explanation(
                explanation,
                X_train[:100],  # Sample for evaluation
                model.predict_proba(X_train[:100])[:, pred_label_idx],
                pred_label_idx,
                model.predict_proba(X_train[:100])[:, pred_label_idx]
            )
            
            metrics['computation_time'] = elapsed_time
            metrics['method'] = method_name
            
            # Get top features
            feature_weights = dict(explanation.as_list(label=pred_label_idx))
            top_features = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:num_features]
            metrics['top_features'] = top_features
            
            results[method_name] = metrics
            
            print(f"  R² Score: {metrics['r2_score']:.4f}")
            print(f"  Computation Time: {elapsed_time:.2f}s")
            print(f"  Top Features: {', '.join([f'{f[0]}: {f[1]:.3f}' for f in top_features[:3]])}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[method_name] = {'error': str(e), 'method': method_name}
    
    return results


def print_comparison_table(results: Dict[str, Dict]):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    # Extract methods and metrics
    methods = list(results.keys())
    metrics_to_show = ['r2_score', 'computation_time', 'num_features', 'max_importance']
    
    # Print header
    header = f"{'Method':<25} {'R² Score':<12} {'Time (s)':<12} {'# Features':<12} {'Max Imp.':<12}"
    print(header)
    print("-" * len(header))
    
    # Print results
    for method in methods:
        if 'error' in results[method]:
            print(f"{method:<25} {'ERROR':<12}")
            continue
        
        r2 = results[method].get('r2_score', 0.0)
        time_val = results[method].get('computation_time', 0.0)
        n_feat = results[method].get('num_features', 0)
        max_imp = results[method].get('max_importance', 0.0)
        
        print(f"{method:<25} {r2:>11.4f} {time_val:>11.2f} {n_feat:>11} {max_imp:>11.4f}")
    
    # Find best method
    valid_results = {k: v for k, v in results.items() if 'error' not in v and 'r2_score' in v}
    if valid_results:
        best_method = max(valid_results.items(), key=lambda x: x[1].get('r2_score', -1))
        print(f"\nBest R² Score: {best_method[0]} ({best_method[1]['r2_score']:.4f})")
        
        # Compare with baseline
        if 'Linear (Baseline)' in valid_results:
            baseline_r2 = valid_results['Linear (Baseline)']['r2_score']
            improvement = best_method[1]['r2_score'] - baseline_r2
            print(f"Improvement over baseline: {improvement:+.4f} ({improvement/baseline_r2*100:+.2f}%)")


def main():
    """Main comparison function."""
    print("="*80)
    print("LIME Explanation Methods Comparison")
    print("="*80)
    
    # Load dataset
    print("\n1. Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    
    # Train model
    print("2. Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    print(f"   Model accuracy: {model.score(X_test, y_test):.4f}")
    
    # Select instance to explain
    test_idx = 0
    instance = X_test[test_idx]
    
    # Compare methods
    print("\n3. Comparing explanation methods...")
    results = compare_methods(
        X_train=X_train,
        y_train=y_train,
        model=model,
        instance=instance,
        feature_names=feature_names,
        class_names=class_names,
        num_features=4,
        num_samples=5000
    )
    
    # Print comparison
    print_comparison_table(results)
    
    # Test with more complex dataset
    print("\n\n" + "="*80)
    print("Testing with more complex synthetic dataset...")
    print("="*80)
    
    X_syn, y_syn = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
        X_syn, y_syn, test_size=0.2, random_state=42
    )
    
    model_syn = GradientBoostingClassifier(n_estimators=100, random_state=0)
    model_syn.fit(X_train_syn, y_train_syn)
    
    instance_syn = X_test_syn[0]
    feature_names_syn = [f"feature_{i}" for i in range(X_syn.shape[1])]
    class_names_syn = [f"class_{i}" for i in range(3)]
    
    results_syn = compare_methods(
        X_train=X_train_syn,
        y_train=y_train_syn,
        model=model_syn,
        instance=instance_syn,
        feature_names=feature_names_syn,
        class_names=class_names_syn,
        num_features=5,
        num_samples=5000
    )
    
    print_comparison_table(results_syn)
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()

