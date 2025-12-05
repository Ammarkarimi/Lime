from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import sys
import uuid
import threading
import subprocess
import time
from pathlib import Path
import re
import numpy as np
import pandas as pd
from io import StringIO
import warnings

# Add repo root to sys.path so we can import lime modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from lime.llm_integration import compile_expression
except Exception as e:
    compile_expression = None
    print(f'Warning: Could not import compile_expression: {e}')

app = Flask(__name__, template_folder='templates')

# Jobs storage: job_id -> {status, output_lines, filename, returncode}
jobs = {}

# Store dataset info for current session
current_dataset_info = {'class_names': None, 'column_value_map': {}, 'column_display_map': {}}


def generate_llm_summary(exp, model, instance, feature_names, class_names=None, pred_label_idx=None):
    """
    Generate a natural language summary of the LIME explanation using LLM.
    
    Args:
        exp: LIME Explanation object
        model: Trained model
        instance: Instance being explained (array)
        feature_names: List of feature names
        class_names: List of class names (for classification)
        pred_label_idx: Predicted class index (for classification)
    
    Returns:
        HTML string with LLM summary or None if generation fails
    """
    try:
        # Check if OpenAI is available
        try:
            import openai
            OPENAI_AVAILABLE = True
        except ImportError:
            OPENAI_AVAILABLE = False
        
        if not OPENAI_AVAILABLE:
            return None
        
        # Get API key from environment or config
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            try:
                # Try to load from .env file
                try:
                    from dotenv import load_dotenv
                    load_dotenv(REPO_ROOT / '.env')
                    api_key = os.getenv('OPENAI_API_KEY')
                except ImportError:
                    pass
                # Try importing from config module
                if not api_key:
                    try:
                        sys.path.insert(0, str(REPO_ROOT))
                        from config import OPENAI_API_KEY as config_key
                        api_key = config_key
                    except:
                        pass
            except:
                pass
        
        if not api_key:
            return None
        
        # Extract feature importance from explanation
        if hasattr(exp, 'as_list'):
            if pred_label_idx is not None:
                # Check if the label exists in the explanation
                try:
                    # Try to get available labels first
                    if hasattr(exp, 'available_labels'):
                        available_labels = exp.available_labels()
                        if pred_label_idx in available_labels:
                            feature_importance = exp.as_list(pred_label_idx)
                        elif len(available_labels) > 0:
                            # Use the first available label if requested label doesn't exist
                            feature_importance = exp.as_list(available_labels[0])
                        else:
                            feature_importance = exp.as_list()
                    else:
                        # Try to get the list, catch exception if label doesn't exist
                        try:
                            feature_importance = exp.as_list(pred_label_idx)
                        except (IndexError, KeyError, ValueError):
                            # Fallback to getting list without label
                            feature_importance = exp.as_list()
                except Exception:
                    # Fallback to getting list without label
                    feature_importance = exp.as_list()
            else:
                feature_importance = exp.as_list()
        else:
            return None
        
        # Get top features (positive and negative)
        top_positive = [(feat, weight) for feat, weight in feature_importance if weight > 0][:5]
        top_negative = [(feat, weight) for feat, weight in feature_importance if weight < 0][:5]
        
        # Get prediction info
        if hasattr(model, 'predict_proba') and pred_label_idx is not None:
            proba = model.predict_proba(instance.reshape(1, -1))[0]
            # Check bounds for pred_label_idx
            if pred_label_idx < len(proba):
                pred_prob = float(proba[pred_label_idx])
            else:
                # Use the maximum probability if index is out of range
                pred_label_idx = int(np.argmax(proba))
                pred_prob = float(proba[pred_label_idx])
            
            if class_names and pred_label_idx < len(class_names):
                pred_class = class_names[pred_label_idx]
            else:
                pred_class = f"Class {pred_label_idx}"
            prediction_info = f"The model predicts {pred_class} with {pred_prob:.2%} confidence."
        else:
            pred = model.predict(instance.reshape(1, -1))[0]
            prediction_info = f"The model predicts a value of {pred:.4f}."
        
        # Build prompt for LLM
        top_features_text = "\n".join([f"- {feat}: {weight:+.4f}" for feat, weight in feature_importance[:10]])
        
        prompt = f"""You are an AI explainability expert. Based on the following LIME (Local Interpretable Model-agnostic Explanations) analysis, provide a clear, concise natural language summary (2-3 sentences) explaining why the model made this prediction.

{prediction_info}

Top contributing features (positive values increase prediction, negative values decrease it):
{top_features_text}

Provide a brief, user-friendly explanation that:
1. Summarizes why the model made this prediction
2. Highlights the most important features and their impact
3. Is written in plain language that non-experts can understand

Keep the response to 2-3 sentences maximum. Do not include markdown formatting."""

        # Call OpenAI API
        try:
            if hasattr(openai, 'OpenAI'):
                # New API (openai >= 1.0)
                client = openai.OpenAI(api_key=api_key, timeout=30)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an AI explainability expert. Provide clear, concise explanations of machine learning model predictions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                summary_text = response.choices[0].message.content.strip()
            else:
                # Old API (openai < 1.0)
                openai.api_key = api_key
                openai.request_timeout = 30
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an AI explainability expert. Provide clear, concise explanations of machine learning model predictions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                summary_text = response['choices'][0]['message']['content'].strip()
            
            # Create HTML for LLM summary
            summary_html = f'''
            <div id="llm_summary" style="display: none; background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #0066cc;">
                <h3 style="margin-top: 0; color: #0066cc; font-size: 18px;">LLM-Generated Summary</h3>
                <p style="margin: 10px 0; line-height: 1.6; color: #333; font-size: 14px;">{summary_text}</p>
            </div>
            '''
            return summary_html
        except Exception as e:
            warnings.warn(f"LLM summary generation failed: {e}")
            return None
    except Exception as e:
        warnings.warn(f"Error generating LLM summary: {e}")
        return None


def inject_llm_summary_into_html(html_path, summary_html):
    """
    Inject LLM summary into the HTML file.
    
    Args:
        html_path: Path to HTML file
        summary_html: HTML string to inject
    """
    if not summary_html:
        return
    
    try:
        # Read the HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Clean the summary HTML to ensure it's properly formatted
        summary_html_clean = summary_html.strip()
        
        # Check if summary already exists
        if '<div id="llm_summary"' in html_content:
            # Replace existing summary - match the entire div including nested content
            pattern = r'<div id="llm_summary"[^>]*>.*?</div>\s*'
            html_content = re.sub(pattern, summary_html_clean + '\n', html_content, flags=re.DOTALL)
        else:
            # Try to insert after the existing summary section
            # Look for the pattern: </div></div> followed by whitespace and <div class="lime top_div"
            # Use string find instead of regex to avoid any potential issues
            search_pattern = '</div>\n        </div>\n        <div class="lime top_div"'
            match_pos = html_content.find(search_pattern)
            if match_pos == -1:
                # Try alternative pattern with different whitespace
                search_pattern = '</div></div>\n        <div class="lime top_div"'
                match_pos = html_content.find(search_pattern)
            if match_pos == -1:
                # Try with minimal whitespace
                search_pattern = '</div></div><div class="lime top_div"'
                match_pos = html_content.find(search_pattern)
            
            if match_pos != -1:
                # Insert the summary HTML right before the top_div
                html_content = html_content[:match_pos] + summary_html_clean + '\n' + html_content[match_pos:]
            else:
                # Fallback: Try to find </body> tag
                body_pos = html_content.find('</body>')
                if body_pos != -1:
                    html_content = html_content[:body_pos] + summary_html_clean + '\n' + html_content[body_pos:]
                else:
                    # If no body tag, append at the end
                    html_content += summary_html_clean
        
        # Write back to file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except Exception as e:
        warnings.warn(f"Failed to inject LLM summary into HTML: {e}")


def run_example_subprocess(job_id: str):
    """Run example_tabular_llm.py as a subprocess and capture output."""
    jobs[job_id]['status'] = 'running'
    jobs[job_id]['output_lines'] = []
    jobs[job_id]['filename'] = None
    cmd = [sys.executable, 'example_tabular_llm.py']
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        jobs[job_id]['proc'] = proc
        # Read lines
        for line in proc.stdout:
            jobs[job_id]['output_lines'].append(line.rstrip())
        proc.wait()
        jobs[job_id]['returncode'] = proc.returncode
        # Look for output file
        expected = REPO_ROOT / 'example_tabular_llm_explanation.html'
        if expected.exists() and proc.returncode == 0:
            jobs[job_id]['status'] = 'done'
            jobs[job_id]['filename'] = expected.name
        else:
            jobs[job_id]['status'] = 'error'
    except Exception as e:
        jobs[job_id]['output_lines'].append(f'Exception: {e}')
        jobs[job_id]['status'] = 'error'


@app.route('/api/builtin_datasets')
def api_builtin_datasets():
    datasets = [
        {'name': 'iris', 'display': 'Iris'},
        {'name': 'wine', 'display': 'Wine'},
        {'name': 'breast_cancer', 'display': 'BreastCancer'}
    ]
    return jsonify({'datasets': datasets})


@app.route('/api/select_dataset', methods=['POST'])
def api_select_dataset():
    data = request.get_json() or {}
    name = data.get('name')
    if not name:
        return jsonify({'error': 'dataset name required'}), 400
    # load dataset
    try:
        class_names = None
        if name == 'iris':
            from sklearn.datasets import load_iris
            ds = load_iris()
            df = pd.DataFrame(ds.data, columns=ds.feature_names)
            # include target column in meta so UI can select it
            df['__target__'] = ds.target
            suggested_target = '__target__'
            suggestion = 'RandomForestClassifier'
            class_names = list(ds.target_names)
        elif name == 'wine':
            from sklearn.datasets import load_wine
            ds = load_wine()
            df = pd.DataFrame(ds.data, columns=ds.feature_names)
            df['__target__'] = ds.target
            suggested_target = '__target__'
            suggestion = 'RandomForestClassifier'
            class_names = list(ds.target_names)
        elif name == 'breast_cancer':
            from sklearn.datasets import load_breast_cancer
            ds = load_breast_cancer()
            df = pd.DataFrame(ds.data, columns=ds.feature_names)
            df['__target__'] = ds.target
            suggested_target = '__target__'
            suggestion = 'RandomForestClassifier'
            class_names = list(ds.target_names)
        else:
            return jsonify({'error': 'unknown dataset'}), 400
        # Compute simple column types
        cols = list(df.columns)
        cols_meta = []
        column_value_map = {}
        for c in cols:
            ser = df[c]
            if pd.api.types.is_integer_dtype(ser):
                t = 'integer'
            elif pd.api.types.is_float_dtype(ser):
                t = 'float'
            elif pd.api.types.is_bool_dtype(ser):
                t = 'boolean'
            else:
                # treat low-cardinality object columns as categorical
                if pd.api.types.is_object_dtype(ser) and ser.nunique() < 20:
                    t = 'categorical'
                else:
                    t = 'string'
            cols_meta.append({'name': c, 'type': t, 'nunique': int(ser.nunique())})
            try:
                nunique = int(ser.nunique())
                if nunique > 0 and nunique <= 100:
                    column_value_map[c] = [str(x) for x in pd.Series(ser.unique()).tolist()]
            except Exception:
                pass

        # Build a display name map for columns (replace __target__ with a friendly name)
        column_display_map = {c: c for c in cols}
        if name == 'iris':
            # replace __target__ with species for clarity
            column_display_map['__target__'] = 'species'
        elif name == 'wine':
            column_display_map['__target__'] = 'class'
        elif name == 'breast_cancer':
            column_display_map['__target__'] = 'diagnosis'

        # Store class_names and display map globally so frontend can use them
        current_dataset_info['class_names'] = class_names
        current_dataset_info['column_display_map'] = column_display_map
        current_dataset_info['column_value_map'] = column_value_map

        return jsonify({'columns': cols, 'columns_meta': cols_meta, 'suggested_target': suggested_target, 'suggested_algorithm': suggestion, 'class_names': class_names, 'column_display_map': column_display_map, 'column_value_map': column_value_map})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_dataset', methods=['POST'])
def api_upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400
    f = request.files['file']
    # Read raw bytes and try to decode. Some CSVs may be encoded in latin-1/windows-1252.
    data_bytes = f.read()
    try:
        text = data_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Try with common single-byte encoding fallback
            text = data_bytes.decode('latin-1')
        except Exception:
            return jsonify({'error': 'Could not decode uploaded file. Please ensure it is UTF-8 or Latin-1 encoded.'}), 400
    # Save to temp CSV
    tmpdir = REPO_ROOT / 'tmp'
    tmpdir.mkdir(exist_ok=True)
    filename = f.filename or f'dataset_{uuid.uuid4().hex}.csv'
    path = tmpdir / filename
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    # read preview
    df = pd.read_csv(StringIO(text))
    # Drop columns that are completely empty (all NaN, empty strings, or whitespace)
    def is_col_empty(col):
        # Treat as empty if all values are NaN, empty string, or whitespace
        return all(pd.isna(x) or (isinstance(x, str) and x.strip() == '') for x in df[col])
    empty_cols = [col for col in df.columns if is_col_empty(col)]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    columns = list(df.columns)
    # compute column types
    cols_meta = []
    column_value_map = {}
    for c in columns:
        ser = df[c]
        if pd.api.types.is_integer_dtype(ser):
            t = 'integer'
        elif pd.api.types.is_float_dtype(ser):
            t = 'float'
        elif pd.api.types.is_bool_dtype(ser):
            t = 'boolean'
        else:
            if pd.api.types.is_object_dtype(ser) and ser.nunique() < 20:
                t = 'categorical'
            else:
                t = 'string'
        cols_meta.append({'name': c, 'type': t, 'nunique': int(ser.nunique())})
        try:
            nunique = int(ser.nunique())
            if nunique > 0 and nunique <= 100:
                column_value_map[c] = [str(x) for x in pd.Series(ser.unique()).tolist()]
        except Exception:
            pass
    # store column value map for uploaded dataset
    current_dataset_info['column_value_map'] = column_value_map
    # no display map for uploads by default
    return jsonify({'path': str(path), 'columns': columns, 'columns_meta': cols_meta, 'column_value_map': column_value_map})


def _suggest_algorithm(y_series: pd.Series):
    # simple heuristic
    if pd.api.types.is_integer_dtype(y_series) or pd.api.types.is_object_dtype(y_series) or y_series.nunique() < 20:
        return 'RandomForestClassifier'
    else:
        return 'RandomForestRegressor'


def run_train_and_explain(job_id: str, params: dict):
    """Train a model on selected dataset and produce LLM-enhanced LIME explanation."""
    jobs[job_id]['status'] = 'running'
    jobs[job_id]['output_lines'] = []
    jobs[job_id]['filename'] = None
    try:
        dataset = params.get('dataset')
        csv_path = params.get('csv_path')
        target = params.get('target')
        algorithm = params.get('algorithm')

        # Load data
        if csv_path:
            df = pd.read_csv(csv_path)
            # Drop all columns that are completely empty (all NaN, empty strings, or whitespace)
            def is_col_empty(col):
                return df[col].apply(lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == '')).all()
            empty_cols = [col for col in df.columns if is_col_empty(col)]
            if empty_cols:
                jobs[job_id]['output_lines'].append(f'Dropping empty columns: {empty_cols}')
                df = df.drop(columns=empty_cols)
            # Drop columns named Unnamed if they are all NaN, empty, or whitespace
            unnamed_cols = [col for col in df.columns if col.startswith('Unnamed') and is_col_empty(col)]
            if unnamed_cols:
                jobs[job_id]['output_lines'].append(f'Dropping Unnamed columns: {unnamed_cols}')
                df = df.drop(columns=unnamed_cols)
        else:
            if dataset == 'iris':
                from sklearn.datasets import load_iris
                ds = load_iris()
                df = pd.DataFrame(ds.data, columns=ds.feature_names)
                df['__target__'] = ds.target
                target = '__target__' if target is None else target
            elif dataset == 'wine':
                from sklearn.datasets import load_wine
                ds = load_wine()
                df = pd.DataFrame(ds.data, columns=ds.feature_names)
                df['__target__'] = ds.target
                target = '__target__' if target is None else target
            elif dataset == 'breast_cancer':
                from sklearn.datasets import load_breast_cancer
                ds = load_breast_cancer()
                df = pd.DataFrame(ds.data, columns=ds.feature_names)
                df['__target__'] = ds.target
                target = '__target__' if target is None else target
            else:
                jobs[job_id]['output_lines'].append(f'Unknown dataset: {dataset}')
                jobs[job_id]['status'] = 'error'
                return

        # Basic preprocessing: drop NA
        jobs[job_id]['output_lines'].append(f'Dataset loaded with {len(df)} rows and {len(df.columns)} columns (target={target})')
        jobs[job_id]['output_lines'].append(f'Dataset head columns: {list(df.columns)[:10]}')
        df = df.dropna()

        if target not in df.columns:
            jobs[job_id]['output_lines'].append(f'Target column {target} not found')
            jobs[job_id]['status'] = 'error'
            return

        # Detect text columns and vectorize them
        X_raw = df.drop(columns=[target])
        y = df[target]
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = X_raw.copy()
        text_cols = [col for col in X_raw.columns if pd.api.types.is_object_dtype(X_raw[col]) or pd.api.types.is_string_dtype(X_raw[col])]
        for col in text_cols:
            # Only vectorize if column is not all numeric/categorical
            if X_raw[col].apply(lambda x: isinstance(x, str)).all():
                jobs[job_id]['output_lines'].append(f'Vectorizing text column: {col}')
                vec = TfidfVectorizer(max_features=100)
                try:
                    X_vec = vec.fit_transform(X_raw[col].fillna("")).toarray()
                    # Create new columns for each TF-IDF feature
                    tfidf_cols = [f"{col}_tfidf_{i}" for i in range(X_vec.shape[1])]
                    X = X.drop(columns=[col])
                    for i, cname in enumerate(tfidf_cols):
                        X[cname] = X_vec[:, i]
                except Exception as e:
                    jobs[job_id]['output_lines'].append(f'Failed to vectorize column {col}: {e}')
        jobs[job_id]['output_lines'].append(f'Using feature columns ({len(X.columns)}): {list(X.columns)[:10]}')

        # Determine problem type / algorithm selection
        # If user provided an algorithm string, use it; otherwise use simple suggestion
        selected_alg = algorithm or _suggest_algorithm(y)
        jobs[job_id]['output_lines'].append(f'Suggested algorithm: {selected_alg}')

        # Simple train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)
        
        # Validate instance_index
        instance_index = int(params.get('instance_index', 0))
        if instance_index < 0 or instance_index >= len(X_test):
            instance_index = 0
            jobs[job_id]['output_lines'].append(f'Warning: instance_index out of range, using index 0 instead')

        # Train model: map UI algorithm names to concrete sklearn classes
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, LinearRegression
        except Exception:
            # fallback imports
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, LinearRegression

        def make_model_from_name(name: str):
            name = (name or '').strip()
            if name == 'RandomForestClassifier':
                return RandomForestClassifier(n_estimators=50, random_state=0)
            if name == 'RandomForestRegressor':
                return RandomForestRegressor(n_estimators=50, random_state=0)
            if name == 'LogisticRegression':
                return LogisticRegression(max_iter=200)
            if name == 'LinearRegression':
                return LinearRegression()
            # fallback heuristics
            if 'Classifier' in name:
                return RandomForestClassifier(n_estimators=50, random_state=0)
            return RandomForestRegressor(n_estimators=50, random_state=0)

        model = make_model_from_name(selected_alg)
        jobs[job_id]['output_lines'].append(f'Instantiated model: {model.__class__.__module__}.{model.__class__.__name__}')
        try:
            model.fit(X_train, y_train)
            jobs[job_id]['output_lines'].append('Model trained')
        except Exception as e:
            jobs[job_id]['output_lines'].append(f'Model.fit failed: {e}')
            # try fallback: if model failed, try a regressor
            try:
                model = RandomForestRegressor(n_estimators=50, random_state=0)
                model.fit(X_train, y_train)
                jobs[job_id]['output_lines'].append('Fallback regressor trained')
            except Exception as e2:
                jobs[job_id]['output_lines'].append(f'Fallback training failed: {e2}')
                raise

        # Choose instance to explain
        instance = X_test[instance_index]

        # Store training data arrays so we can re-use for later instance explanations
        jobs[job_id]['X_train'] = X_train
        jobs[job_id]['y_train'] = y_train

        # Create Lime explainer
        from lime.lime_tabular import LimeTabularExplainer
        feature_names = list(X.columns)
        # Choose mode based on whether model provides predict_proba (classification)
        explainer_mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'
        jobs[job_id]['output_lines'].append(f'Creating Lime explainer in mode={explainer_mode}')
        
        # Get class names and ensure they match the number of classes
        class_names = current_dataset_info.get('class_names')
        if explainer_mode == 'classification':
            # If class_names is not set or doesn't match, generate them
            if class_names is None:
                # Get unique classes from target
                unique_classes = sorted(y.unique()) if hasattr(y, 'unique') else sorted(np.unique(y))
                num_classes = len(unique_classes)
                class_names = [f'Class_{i}' for i in range(num_classes)]
                jobs[job_id]['output_lines'].append(f'Generated class names: {class_names}')
            else:
                # Ensure class_names has enough elements
                if hasattr(model, 'predict_proba'):
                    num_classes = model.predict_proba(X_train[:1])[0].shape[0]
                    if len(class_names) < num_classes:
                        # Extend class_names if needed
                        class_names = list(class_names) + [f'Class_{i}' for i in range(len(class_names), num_classes)]
                        jobs[job_id]['output_lines'].append(f'Extended class names to match model: {class_names}')
        
        explainer = LimeTabularExplainer(
            training_data=X_train,
            mode=explainer_mode,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
            random_state=0
        )

        # Try LLM integration if available
        try:
            from lime.llm_integration import generate_llm_expression, compile_expression
            LLM_OK = True
        except Exception:
            LLM_OK = False

        pred_label_idx = 0
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(instance.reshape(1, -1))[0]
            pred_label_idx = int(np.argmax(proba))
            # Use class names if available
            class_names = current_dataset_info.get('class_names')
            if class_names and pred_label_idx < len(class_names):
                class_name = class_names[pred_label_idx]
                jobs[job_id]['output_lines'].append(f'Model predicts class {class_name} (index={pred_label_idx}) with prob {proba[pred_label_idx]:.4f}')
            else:
                jobs[job_id]['output_lines'].append(f'Model predicts class index {pred_label_idx} with prob {proba[pred_label_idx]:.4f}')
        else:
            pred = model.predict(instance.reshape(1, -1))[0]
            jobs[job_id]['output_lines'].append(f'Model predicts value {pred}')

        # Compute training metrics on test set
        try:
            from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                acc = accuracy_score(y_test, y_pred)
                jobs[job_id]['metrics'] = {'accuracy': float(acc)}
                jobs[job_id]['output_lines'].append(f'Training accuracy (test set): {acc:.4f}')
            else:
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                jobs[job_id]['metrics'] = {'r2': float(r2), 'mse': float(mse)}
                jobs[job_id]['output_lines'].append(f'Test R2: {r2:.4f}, MSE: {mse:.4f}')
        except Exception:
            jobs[job_id]['metrics'] = None

        # Generate expression using LLM if possible
        expr = None
        if LLM_OK and os.getenv('OPENAI_API_KEY'):
            try:
                jobs[job_id]['output_lines'].append('Attempting to generate LLM expression...')
                # For classifiers, use probability for predicted class as regression target
                if hasattr(model, 'predict_proba'):
                    # Ensure pred_label_idx is within bounds
                    all_proba = model.predict_proba(X_train)
                    if pred_label_idx >= all_proba.shape[1]:
                        # Use the class with highest average probability
                        pred_label_idx = int(np.argmax(all_proba.mean(axis=0)))
                        jobs[job_id]['output_lines'].append(f'Adjusted pred_label_idx to {pred_label_idx} (was out of range)')
                    y_train_proba = all_proba[:, pred_label_idx]
                    expr = generate_llm_expression(X_train, y_train_proba, feature_names=feature_names)
                else:
                    expr = generate_llm_expression(X_train, model.predict(X_train), feature_names=feature_names)
                if expr:
                    jobs[job_id]['output_lines'].append('Generated expression:')
                    jobs[job_id]['output_lines'].append(expr)
                else:
                    jobs[job_id]['output_lines'].append('LLM did not return a valid expression; falling back.')
            except Exception as e:
                jobs[job_id]['output_lines'].append(f'LLM generation failed: {e}')

        # Use LLMEnhancedLimeExplainer
        from lime.llm_lime_wrapper import LLMEnhancedLimeExplainer
        llm_explainer = LLMEnhancedLimeExplainer(
            base_explainer=explainer,
            use_llm_model=True,
            importance_method='gradient',
            llm_provider='openai',
            llm_model=None,
            use_simple_polynomial=True,
            max_degree=3,
            random_state=0
        )

        # Generate explanation - handle potential label issues
        try:
            # For classification, try to explain the predicted label
            # But if that fails, use available labels
            if hasattr(model, 'predict_proba'):
                # Check if we should use a specific label or let LIME choose
                labels_to_explain = (pred_label_idx,)
                try:
                    exp = llm_explainer.explain_instance(
                        data_row=instance,
                        predict_fn=model.predict_proba,
                        labels=labels_to_explain,
                        num_features=min(10, len(feature_names)),
                        num_samples=500
                    )
                except (IndexError, KeyError, ValueError) as e:
                    # If explaining specific label fails, let LIME choose
                    jobs[job_id]['output_lines'].append(f'Warning: Could not explain label {pred_label_idx}, using default labels: {e}')
                    exp = llm_explainer.explain_instance(
                        data_row=instance,
                        predict_fn=model.predict_proba,
                        labels=None,  # Let LIME choose
                        num_features=min(10, len(feature_names)),
                        num_samples=500
                    )
            else:
                exp = llm_explainer.explain_instance(
                    data_row=instance,
                    predict_fn=model.predict,
                    labels=None,
                    num_features=min(10, len(feature_names)),
                    num_samples=500
                )
        except Exception as e:
            jobs[job_id]['output_lines'].append(f'Error generating explanation: {e}')
            raise

        # Save model and expression into the job record for manual evaluation
        jobs[job_id]['model'] = model
        jobs[job_id]['expr'] = expr
        jobs[job_id]['feature_names'] = feature_names
        # Store the corrected class_names (the ones we used for the explainer)
        jobs[job_id]['class_names'] = class_names
        # Store a sample instance (dict) for UI suggested testing
        try:
            sample_vals = {feature_names[i]: float(instance[i]) for i in range(len(feature_names))}
            jobs[job_id]['sample_instance'] = sample_vals
        except Exception:
            jobs[job_id]['sample_instance'] = None

        outname = f'train_explanation_{job_id}.html'
        outpath = REPO_ROOT / outname
        
        # Ensure explanation has proper class_names set
        if hasattr(exp, 'class_names') and exp.class_names is None:
            exp.class_names = class_names
        elif hasattr(exp, 'class_names') and class_names is not None:
            # Ensure class_names length matches
            if hasattr(model, 'predict_proba'):
                num_classes = model.predict_proba(X_train[:1])[0].shape[0]
                if len(exp.class_names) < num_classes:
                    exp.class_names = class_names if len(class_names) >= num_classes else list(exp.class_names) + [f'Class_{i}' for i in range(len(exp.class_names), num_classes)]
        
        # Determine which labels to save - check if pred_label_idx exists in explanation
        labels_to_save = None
        if hasattr(model, 'predict_proba'):
            try:
                # Check if the label exists in the explanation
                if hasattr(exp, 'available_labels'):
                    available_labels = exp.available_labels()
                    if available_labels and len(available_labels) > 0:
                        if pred_label_idx in available_labels:
                            labels_to_save = (pred_label_idx,)
                        else:
                            # Use first available label
                            labels_to_save = (available_labels[0],)
                            jobs[job_id]['output_lines'].append(f'Using available label {available_labels[0]} instead of {pred_label_idx}')
                    else:
                        # No available labels, let LIME decide
                        labels_to_save = None
                else:
                    # Try to save with pred_label_idx, but validate it first
                    # Check if pred_label_idx is valid
                    if hasattr(exp, 'local_exp') and pred_label_idx in exp.local_exp:
                        labels_to_save = (pred_label_idx,)
                    else:
                        labels_to_save = None
            except Exception as e:
                jobs[job_id]['output_lines'].append(f'Error checking available labels: {e}')
                # If checking fails, let LIME decide
                labels_to_save = None
        
        # Try to save with error handling
        max_retries = 3
        saved = False
        for attempt in range(max_retries):
            try:
                if attempt == 0 and labels_to_save is not None:
                    # First try with specific labels
                    exp.save_to_file(str(outpath), labels=labels_to_save)
                elif attempt == 1:
                    # Second try: get available labels and use first one
                    if hasattr(exp, 'available_labels'):
                        try:
                            avail = exp.available_labels()
                            if avail and len(avail) > 0:
                                exp.save_to_file(str(outpath), labels=(avail[0],))
                            else:
                                exp.save_to_file(str(outpath), labels=None)
                        except:
                            exp.save_to_file(str(outpath), labels=None)
                    else:
                        exp.save_to_file(str(outpath), labels=None)
                else:
                    # Final try: save without labels (let LIME decide)
                    exp.save_to_file(str(outpath), labels=None)
                saved = True
                break
            except (IndexError, KeyError, ValueError) as e:
                if attempt < max_retries - 1:
                    jobs[job_id]['output_lines'].append(f'Warning: Save attempt {attempt + 1} failed: {e}, retrying...')
                    labels_to_save = None  # Reset for next attempt
                else:
                    jobs[job_id]['output_lines'].append(f'Error saving explanation after {max_retries} attempts: {e}')
                    raise
            except Exception as e:
                jobs[job_id]['output_lines'].append(f'Unexpected error saving explanation: {e}')
                raise
        
        if not saved:
            raise Exception("Failed to save explanation after all retry attempts")
        
        # Generate and inject LLM summary
        try:
            jobs[job_id]['output_lines'].append('Generating LLM summary...')
            class_names = current_dataset_info.get('class_names')
            # Determine which label to use for summary - use the label that was actually explained
            label_for_summary = None
            if hasattr(model, 'predict_proba'):
                # Check which label was actually explained
                if hasattr(exp, 'available_labels'):
                    available_labels = exp.available_labels()
                    if pred_label_idx in available_labels:
                        label_for_summary = pred_label_idx
                    elif len(available_labels) > 0:
                        label_for_summary = available_labels[0]
                else:
                    label_for_summary = pred_label_idx
            
            summary_html = generate_llm_summary(
                exp, model, instance, feature_names, 
                class_names=class_names, 
                pred_label_idx=label_for_summary
            )
            if summary_html:
                inject_llm_summary_into_html(str(outpath), summary_html)
                jobs[job_id]['output_lines'].append('LLM summary generated and injected successfully')
            else:
                jobs[job_id]['output_lines'].append('LLM summary generation skipped (API key not available or generation failed)')
        except Exception as e:
            jobs[job_id]['output_lines'].append(f'LLM summary generation error: {e}')
        
        jobs[job_id]['filename'] = outname
        jobs[job_id]['status'] = 'done'
        jobs[job_id]['output_lines'].append(f'Saved explanation HTML to {outpath}')
    except Exception as e:
        jobs[job_id]['output_lines'].append(f'Exception during train/explain: {e}')
        jobs[job_id]['status'] = 'error'


@app.route('/api/train_and_explain', methods=['POST'])
def api_train_and_explain():
    params = request.get_json() or {}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'queued', 'output_lines': [], 'filename': None, 'returncode': None}
    thread = threading.Thread(target=run_train_and_explain, args=(job_id, params), daemon=True)
    thread.start()
    return jsonify({'job_id': job_id}), 202


@app.route('/')
def index():
    datasets = [
        {'name': 'iris', 'display': 'Iris'},
        {'name': 'wine', 'display': 'Wine'},
        {'name': 'breast_cancer', 'display': 'BreastCancer'}
    ]
    return render_template('index.html', datasets=datasets)


@app.route('/api/run', methods=['POST'])
def api_run():
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'queued', 'output_lines': [], 'filename': None, 'returncode': None}
    thread = threading.Thread(target=run_example_subprocess, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({'job_id': job_id}), 202


@app.route('/api/status/<job_id>')
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'job not found'}), 404
    # Return last 2000 chars of logs
    logs = '\n'.join(job['output_lines'][-200:])
    return jsonify({'status': job['status'], 'logs': logs, 'filename': job.get('filename'), 'metrics': job.get('metrics'), 'sample_instance': job.get('sample_instance'), 'feature_names': job.get('feature_names')})


@app.route('/api/latest_result')
def api_latest_result():
    """Return the filename of the most recent completed job result, if any."""
    # Search jobs for the most recent job with a filename
    for job_id in reversed(list(jobs.keys())):
        job = jobs.get(job_id)
        if job and job.get('filename') and job.get('status') == 'done':
            return jsonify({'filename': job.get('filename')})

    # Fallback: look for files matching train_explanation_*.html in REPO_ROOT
    try:
        candidates = list(REPO_ROOT.glob('train_explanation_*.html'))
        candidates += list(REPO_ROOT.glob('example_tabular_llm_explanation.html'))
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            return jsonify({'filename': latest.name})
    except Exception:
        pass

    return jsonify({'filename': None})


@app.route('/api/explain_instance', methods=['POST'])
def api_explain_instance():
    """Explain a provided instance using the most recent trained model/job.
    Expects JSON: feature_name -> value map. Optional 'job_id' to select a job.
    Returns: filename of generated explanation HTML and model prediction.
    """
    data = request.get_json() or {}
    job_id = data.get('job_id')
    target_job = None
    if job_id:
        target_job = jobs.get(job_id)
    else:
        # find most recent done job with model and training data
        for j in reversed(list(jobs.values())):
            if j.get('status') == 'done' and j.get('model') is not None and j.get('X_train') is not None:
                target_job = j
                break

    if not target_job:
        return jsonify({'error': 'No trained job with model found. Run Train & Explain first.'}), 400

    model = target_job.get('model')
    feature_names = target_job.get('feature_names')
    X_train = target_job.get('X_train')

    # Build feature vector
    try:
        vals = [float(data.get(fname)) for fname in feature_names]
    except Exception:
        return jsonify({'error': f'Provide numerical values for all features: {feature_names}'}), 400

    X_row = np.array([vals])

    # Predict
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_row)[0]
            pred_idx = int(np.argmax(proba))
            pred = {'index': pred_idx, 'probability': float(proba[pred_idx])}
        else:
            p = model.predict(X_row)[0]
            pred = {'value': float(p)}
            pred_idx = None
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    # Create explainer and explain this instance
    try:
        from lime.lime_tabular import LimeTabularExplainer
        from lime.llm_lime_wrapper import LLMEnhancedLimeExplainer

        class_names = target_job.get('class_names')
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=class_names,
            mode='classification' if hasattr(model, 'predict_proba') else 'regression',
            discretize_continuous=True,
            random_state=0
        )

        llm_explainer = LLMEnhancedLimeExplainer(
            base_explainer=explainer,
            use_llm_model=True,
            importance_method='gradient',
            llm_provider='openai',
            llm_model=None,
            use_simple_polynomial=True,
            max_degree=3,
            random_state=0
        )

        exp = llm_explainer.explain_instance(
            data_row=np.array(vals),
            predict_fn=model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
            labels=(pred_idx,) if hasattr(model, 'predict_proba') else None,
            num_features=min(10, len(feature_names)),
            num_samples=500
        )

        outname = f'explain_instance_{uuid.uuid4().hex}.html'
        outpath = REPO_ROOT / outname
        exp.save_to_file(str(outpath), labels=(pred_idx,) if hasattr(model, 'predict_proba') else None)
        
        # Generate and inject LLM summary
        try:
            class_names = target_job.get('class_names')
            summary_html = generate_llm_summary(
                exp, model, np.array(vals), feature_names,
                class_names=class_names,
                pred_label_idx=pred_idx if hasattr(model, 'predict_proba') else None
            )
            if summary_html:
                inject_llm_summary_into_html(str(outpath), summary_html)
        except Exception as e:
            # Don't fail the request if summary generation fails
            pass
        
        return jsonify({'filename': outname, 'prediction': pred})
    except Exception as e:
        return jsonify({'error': f'Explanation failed: {e}'}), 500


@app.route('/result')
def result():
    filename = request.args.get('filename')
    if not filename:
        return redirect(url_for('index'))
    file_path = REPO_ROOT / filename
    if not file_path.exists():
        return render_template('index.html', error='Result file not found: ' + filename)
    return render_template('result.html', result_path=filename)


@app.route('/results/<path:filename>')
def serve_result_file(filename):
    # Serve file from repo root
    return send_from_directory(str(REPO_ROOT), filename)


def _extract_expression_from_logs(logs: str) -> str:
    """Extract the LLM-generated expression from logs text.
    Looks for a line starting with 'Generated expression:' and returns the next non-empty line,
    or the remainder of that line if present.
    """
    if not logs:
        return None
    # Try to find 'Generated expression:'
    idx = logs.find('Generated expression:')
    if idx == -1:
        # Try lowercase
        idx = logs.find('generated expression:')
    if idx == -1:
        return None
    tail = logs[idx:]
    # Split lines
    lines = tail.splitlines()
    if len(lines) == 1:
        # single line, try regex
        m = re.search(r'Generated expression:\s*(.+)', logs, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None
    # If first line contains colon and nothing else, take next non-empty line
    first = lines[0]
    after = '\n'.join(lines[1:])
    for line in after.splitlines():
        if line.strip():
            return line.strip()
    # fallback: regex on whole logs
    m = re.search(r'Generated expression:\s*(.+)', logs, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """Evaluate user-provided feature values against the latest LLM expression and model.
    Expects JSON: {sepal_length, sepal_width, petal_length, petal_width}
    """
    data = request.get_json() or {}

    # Find the most recent completed job that has a stored model (trained via Train & Explain)
    latest_job = None
    for job in reversed(list(jobs.values())):
        if job.get('status') == 'done' and job.get('model') is not None:
            latest_job = job
            break

    if latest_job is None:
        return jsonify({'error': 'No trained model found. Please run Train & Explain first.'}), 400

    model = latest_job.get('model')
    feature_names = latest_job.get('feature_names') or []
    expr = latest_job.get('expr')

    # Expect data to be a map of feature_name -> value
    try:
        # Build feature vector in the same order as feature_names
        vals = [float(data.get(fn)) for fn in feature_names]
    except Exception:
        return jsonify({'error': f'Invalid input. Provide values for features: {feature_names}'}), 400

    X_row = np.array([vals])

    # Model prediction
    model_pred = None
    model_proba = None
    model_pred_name = None
    try:
        if hasattr(model, 'predict_proba'):
            proba_arr = model.predict_proba(X_row)[0]
            idx = int(np.argmax(proba_arr))
            model_pred = idx
            model_proba = float(proba_arr[idx])
            # Try to map class name if available
            if 'class_names' in latest_job and latest_job['class_names'] is not None:
                names = latest_job['class_names']
                if idx < len(names):
                    model_pred_name = names[idx]
        else:
            p = model.predict(X_row)[0]
            model_pred = float(p)
    except Exception as e:
        return jsonify({'error': f'Model prediction failed: {e}'}), 500

    # Evaluate LLM expression if available
    llm_value = None
    llm_bool = None
    expr_used = None
    if expr and compile_expression is not None:
        try:
            predict_fn = compile_expression(expr, feature_names)
            llm_pred = predict_fn(X_row)
            if hasattr(llm_pred, '__len__'):
                llm_value = float(llm_pred[0])
            else:
                llm_value = float(llm_pred)
            llm_bool = bool(llm_value)
            expr_used = expr
        except Exception as e:
            llm_value = None
            llm_bool = None

    # Post-process raw LLM value into a probability-like score using sigmoid
    llm_probability = None
    try:
        if llm_value is not None:
            # Use numpy exp (np already imported) and coerce to float
            llm_probability = float(1.0 / (1.0 + float(np.exp(-llm_value))))
    except Exception:
        llm_probability = None

    return jsonify({
        'model_prediction': model_pred_name if model_pred_name is not None else model_pred,
        'model_probability': model_proba,
        'llm_expression': expr_used,
        'llm_value': llm_value,
        'llm_probability': llm_probability,
        'llm_bool': llm_bool,
        'features': feature_names
    })


if __name__ == '__main__':
    # Default host and port
    app.run(host='127.0.0.1', port=5000, debug=True)
