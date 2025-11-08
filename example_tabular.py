import os
from typing import List

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lime.lime_tabular import LimeTabularExplainer


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train a simple classifier for the demo."""
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X, y)
    return model


def explain_instance(
    explainer: LimeTabularExplainer,
    model: RandomForestClassifier,
    instance: np.ndarray,
    class_names: List[str],
    num_features: int = 5,
):
    """Generate a LIME explanation for a single instance and return objects useful for printing/saving."""
    predict_proba = model.predict_proba
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_proba,
        num_features=num_features,
        top_labels=1,
    )
    pred_proba = predict_proba(instance.reshape(1, -1))[0]
    pred_label_index = int(np.argmax(pred_proba))
    pred_label_name = class_names[pred_label_index]
    return exp, pred_label_index, pred_label_name, pred_proba[pred_label_index]


def main() -> None:
    iris = load_iris()
    X: np.ndarray = iris.data
    y: np.ndarray = iris.target
    feature_names: List[str] = list(iris.feature_names)
    class_names: List[str] = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    model = train_model(X_train, y_train)
    test_idx = 0
    instance = X_test[test_idx]

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=0,
    )

    exp, label_idx, label_name, label_prob = explain_instance(
        explainer=explainer,
        model=model,
        instance=instance,
        class_names=class_names,
        num_features=min(6, X_train.shape[1]),
    )

    print("Model prediction:")
    print(f"  class = {label_name} (index={label_idx})")
    print(f"  probability = {label_prob:.4f}")
    print()
    print("Top contributing features (feature, contribution):")
    for feat, weight in exp.as_list(label=label_idx):
        print(f"  {feat:40s} {weight:+.4f}")

    # Save an interactive HTML explanation next to this script
    out_dir = os.path.join(os.path.dirname(__file__), "_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "example_tabular_iris.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(exp.as_html(labels=[label_idx]))
    print()
    print(f"Saved HTML explanation to: {out_path}")


if __name__ == "__main__":
    main()


