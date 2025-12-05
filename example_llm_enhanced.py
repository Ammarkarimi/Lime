"""
Example demonstrating LLM-enhanced LIME for non-linear local approximations.

This example shows how to use LLM-generated non-linear models with LIME
to get better explanations when linear models fail.
"""

import os
from typing import List

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from lime.lime_tabular import LimeTabularExplainer
from lime.llm_lime_wrapper import LLMNonLinearRegressor, LLMEnhancedLimeExplainer
from lime.evaluation_metrics import ExplanationEvaluator


def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf') -> object:
    """Train a classifier for the demo."""
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=200, random_state=0)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(n_estimators=100, random_state=0)
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X, y)
    return model


def compare_explanations(
    explainer_linear,
    explainer_llm,
    model,
    instance: np.ndarray,
    class_names: List[str],
    label_idx: int,
    num_features: int = 5
):
    """Compare linear vs LLM-enhanced explanations."""
    print("\n" + "="*80)
    print("COMPARING LINEAR vs LLM-ENHANCED LIME EXPLANATIONS")
    print("="*80)
    
    # Linear explanation
    exp_linear = explainer_linear.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=num_features,
        top_labels=1,
    )
    
    # LLM-enhanced explanation
    exp_llm = explainer_llm.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=num_features,
        top_labels=1,
    )
    
    print(f"\nPredicted class: {class_names[label_idx]}")
    print(f"\n{'Feature':<40} {'Linear Weight':<15} {'LLM Weight':<15} {'Difference':<15}")
    print("-" * 85)
    
    linear_weights = dict(exp_linear.as_list(label=label_idx))
    llm_weights = dict(exp_llm.as_list(label=label_idx))
    
    all_features = set(linear_weights.keys()) | set(llm_weights.keys())
    
    for feat in sorted(all_features, key=lambda x: abs(llm_weights.get(x, 0)), reverse=True):
        linear_w = linear_weights.get(feat, 0.0)
        llm_w = llm_weights.get(feat, 0.0)
        diff = llm_w - linear_w
        print(f"{feat:<40} {linear_w:>+14.4f} {llm_w:>+14.4f} {diff:>+14.4f}")
    
    # Compare scores
    print(f"\nModel Fit Scores:")
    print(f"  Linear R²:  {exp_linear.score[label_idx]:.4f}")
    print(f"  LLM R²:     {exp_llm.score[label_idx]:.4f}")
    print(f"  Improvement: {exp_llm.score[label_idx] - exp_linear.score[label_idx]:+.4f}")
    
    return exp_linear, exp_llm


def main() -> None:
    """Main demonstration function."""
    print("LLM-Enhanced LIME Demonstration")
    print("="*80)
    
    # Load dataset
    print("\n1. Loading Iris dataset...")
    iris = load_iris()
    X: np.ndarray = iris.data
    y: np.ndarray = iris.target
    feature_names: List[str] = list(iris.feature_names)
    class_names: List[str] = list(iris.target_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    
    # Train a non-linear model (Random Forest)
    print("2. Training Random Forest classifier (non-linear model)...")
    model = train_model(X_train, y_train, model_type='rf')
    test_idx = 0
    instance = X_test[test_idx]
    
    pred_proba = model.predict_proba(instance.reshape(1, -1))[0]
    pred_label_index = int(np.argmax(pred_proba))
    pred_label_name = class_names[pred_label_index]
    
    print(f"   Instance prediction: {pred_label_name} (prob={pred_proba[pred_label_index]:.4f})")
    
    # Create standard LIME explainer
    print("\n3. Creating standard LIME explainer (linear model)...")
    explainer_linear = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=0,
    )
    
    # Create LLM-enhanced explainer
    print("4. Creating LLM-enhanced LIME explainer (non-linear model)...")
    
    # Check if API key is available
    from lime.config import OPENAI_API_KEY
    llm_provider = 'openai' if OPENAI_API_KEY else 'none'
    
    if llm_provider == 'none':
        print("   Note: No OpenAI API key found. Using polynomial features as fallback.")
        print("   Set OPENAI_API_KEY environment variable to use real LLM integration.")
    
    explainer_llm = LLMEnhancedLimeExplainer(
        base_explainer=explainer_linear,
        use_llm_model=True,
        importance_method='gradient',  # or 'permutation'
        llm_provider=llm_provider,
        llm_model='gpt-4o-mini',  # Cost-effective model
        use_simple_polynomial=True,  # Fallback if LLM fails
        max_degree=3,
    )
    
    # Compare explanations
    num_features = min(4, X_train.shape[1])
    exp_linear, exp_llm = compare_explanations(
        explainer_linear=explainer_linear,
        explainer_llm=explainer_llm,
        model=model,
        instance=instance,
        class_names=class_names,
        label_idx=pred_label_index,
        num_features=num_features
    )
    
    # Additional evaluation metrics
    print("\n6. Computing additional evaluation metrics...")
    evaluator = ExplanationEvaluator()
    
    # Get neighborhood data for evaluation (simplified - in practice, get from explainer)
    X_sample = X_train[:100]
    y_sample = model.predict_proba(X_sample)[:, pred_label_index]
    
    metrics_linear = evaluator.evaluate_explanation(
        exp_linear, X_sample, y_sample, pred_label_index, y_sample
    )
    metrics_llm = evaluator.evaluate_explanation(
        exp_llm, X_sample, y_sample, pred_label_index, y_sample
    )
    
    print(f"\nDetailed Metrics:")
    print(f"  Linear - R²: {metrics_linear['r2_score']:.4f}, MSE: {metrics_linear.get('mse', 0):.4f}")
    print(f"  LLM    - R²: {metrics_llm['r2_score']:.4f}, MSE: {metrics_llm.get('mse', 0):.4f}")
    
    comparison = evaluator.compare_explanations(
        exp_linear, exp_llm, X_sample, y_sample, pred_label_index, y_sample
    )
    print(f"\nComparison:")
    print(f"  R² Improvement: {comparison['r2_improvement']:+.4f}")
    print(f"  Rank Correlation: {comparison['rank_correlation']:.4f}")
    print(f"  Top-5 Feature Overlap: {comparison['top_k_overlap']:.2%}")
    
    # Save HTML outputs
    print("\n5. Saving HTML explanations...")
    out_dir = os.path.join(os.path.dirname(__file__), "_outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path_linear = os.path.join(out_dir, "example_linear_iris.html")
    with open(out_path_linear, "w", encoding="utf-8") as f:
        f.write(exp_linear.as_html(labels=[pred_label_index]))
    print(f"   Saved linear explanation to: {out_path_linear}")
    
    out_path_llm = os.path.join(out_dir, "example_llm_enhanced_iris.html")
    with open(out_path_llm, "w", encoding="utf-8") as f:
        f.write(exp_llm.as_html(labels=[pred_label_index]))
    print(f"   Saved LLM-enhanced explanation to: {out_path_llm}")
    
    print("\n" + "="*80)
    print("Demonstration complete!")
    print("="*80)
    
    # Additional example with more complex dataset
    print("\n\n6. Testing with more complex synthetic dataset...")
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
    
    model_syn = train_model(X_train_syn, y_train_syn, model_type='gb')
    instance_syn = X_test_syn[0]
    
    explainer_linear_syn = LimeTabularExplainer(
        training_data=X_train_syn,
        mode="classification",
        random_state=0,
    )
    
    explainer_llm_syn = LLMEnhancedLimeExplainer(
        base_explainer=explainer_linear_syn,
        use_llm_model=True,
        importance_method='gradient',
    )
    
    exp_linear_syn = explainer_linear_syn.explain_instance(
        data_row=instance_syn,
        predict_fn=model_syn.predict_proba,
        num_features=5,
        top_labels=1,
    )
    
    exp_llm_syn = explainer_llm_syn.explain_instance(
        data_row=instance_syn,
        predict_fn=model_syn.predict_proba,
        num_features=5,
        top_labels=1,
    )
    
    pred_label_syn = int(np.argmax(model_syn.predict_proba(instance_syn.reshape(1, -1))[0]))
    
    print(f"\nSynthetic dataset results:")
    print(f"  Linear R²:  {exp_linear_syn.score[pred_label_syn]:.4f}")
    print(f"  LLM R²:     {exp_llm_syn.score[pred_label_syn]:.4f}")
    print(f"  Improvement: {exp_llm_syn.score[pred_label_syn] - exp_linear_syn.score[pred_label_syn]:+.4f}")


if __name__ == "__main__":
    main()

