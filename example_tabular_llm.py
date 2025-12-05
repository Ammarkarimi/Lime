"""
Example: Use LLM to generate a local symbolic model and use it with LIME.

This script trains a classifier on Iris, selects an instance, then:
 1. Attempts to generate a symbolic expression for the model's prediction
    on a chosen class using the configured LLM provider (OpenAI/Anthropic).
 2. Compiles the expression into a callable and evaluates it on the instance.
 3. Uses `LLMEnhancedLimeExplainer` to produce an explanation that uses
    the LLM-based regressor (falls back to polynomial features if LLM fails).

Usage:
  - Ensure your OpenAI key is available: set environment variable `OPENAI_API_KEY`.
  - Run: `python example_tabular_llm.py`

Note: LLM calls consume API credits; the script will fall back to polynomial
      approximations when LLM is unavailable or fails.
"""

import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lime.lime_tabular import LimeTabularExplainer
from lime.llm_lime_wrapper import LLMEnhancedLimeExplainer

try:
    from lime.llm_integration import generate_llm_expression, compile_expression
    LLM_INTEGRATION_AVAILABLE = True
except Exception:
    LLM_INTEGRATION_AVAILABLE = False


def main():
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    model = RandomForestClassifier(n_estimators=50, random_state=0)
    model.fit(X_train, y_train)

    instance = X_test[0]

    # Create base explainer
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=list(data.feature_names),
        class_names=list(data.target_names),
        discretize_continuous=True,
        random_state=0
    )

    # Predict to get class index
    proba = model.predict_proba(instance.reshape(1, -1))[0]
    pred_label_idx = int(np.argmax(proba))
    print(f"Model predicts class {data.target_names[pred_label_idx]} with prob {proba[pred_label_idx]:.4f}")

    # --- Attempt to generate LLM expression for the predicted class ---
    if LLM_INTEGRATION_AVAILABLE and os.getenv('OPENAI_API_KEY'):
        print("Attempting to generate LLM expression (this will call the API)...")
        # Use training data predictions for the target class as the regression target
        y_train_proba = model.predict_proba(X_train)[:, pred_label_idx]
        expr = generate_llm_expression(X_train, y_train_proba, feature_names=list(data.feature_names), provider='openai')
        if expr:
            print("Generated expression:")
            print(expr)
            fn = compile_expression(expr, list(data.feature_names))
            pred_from_expr = fn(instance.reshape(1, -1))[0] if hasattr(fn, '__call__') else fn(instance.reshape(1, -1))[0]
            print(f"LLM expression prediction on the instance: {pred_from_expr:.4f}")
        else:
            print("LLM did not return a valid expression; falling back to polynomial approximation.")
    else:
        print("LLM integration not available or OPENAI_API_KEY not set; skipping expression generation.")

    # --- Use LLMEnhancedLimeExplainer to produce an explanation ---
    print("Running LLM-enhanced LIME explainer (may fall back to polynomial regressor)...")
    llm_explainer = LLMEnhancedLimeExplainer(
        base_explainer=explainer,
        use_llm_model=True,
        importance_method='gradient',
        llm_provider='openai',
        llm_model=None,  # use default from config
        use_simple_polynomial=True,  # fallback if LLM fails
        max_degree=3,
        random_state=0
    )

    exp = llm_explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        labels=(pred_label_idx,),
        num_features=4,
        num_samples=500  # reduce samples for speed in example
    )

    print("Top explanation features:")
    print(exp.as_list(label=pred_label_idx))

    # Save HTML visualization
    outpath = 'example_tabular_llm_explanation.html'
    exp.save_to_file(outpath, labels=(pred_label_idx,))
    print(f"Saved explanation HTML to {outpath}")


if __name__ == '__main__':
    main()
