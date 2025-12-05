#!/usr/bin/env python
"""Test script to verify the HTML summary section is generated."""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

# Load iris dataset
ds = load_iris()
X = ds.data
y = ds.target

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=0)
model.fit(X, y)

# Create explainer
explainer = LimeTabularExplainer(
    X,
    feature_names=ds.feature_names,
    mode='classification',
    discretize_continuous=True,
    random_state=0
)

# Explain first instance
exp = explainer.explain_instance(
    X[0],
    model.predict_proba,
    num_features=4
)

# Save to file
outfile = 'test_summary_output.html'
exp.save_to_file(outfile)
print(f'Saved explanation to {outfile}')

# Read and check for summary section
with open(outfile, 'r', encoding='utf8') as f:
    content = f.read()
    if '<div class="summary">' in content:
        print('✓ Summary section found in HTML!')
        if 'Model prediction:' in content:
            print('✓ Model prediction found in summary!')
        if 'Top contributing features:' in content:
            print('✓ Top contributing features found in summary!')
    else:
        print('✗ Summary section NOT found in HTML')
