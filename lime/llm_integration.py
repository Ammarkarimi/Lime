"""
LLM API Integration for generating symbolic expressions.

Supports OpenAI and Anthropic APIs for generating mathematical expressions
that approximate local model behavior.
"""

import os
import re
import hashlib
import json
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv
import numpy as np
import warnings

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("openai package not installed. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY,
    DEFAULT_LLM_MODEL, DEFAULT_ANTHROPIC_MODEL,
    MAX_RETRIES, TIMEOUT_SECONDS, CACHE_RESPONSES
)


# Simple in-memory cache (can be replaced with Redis, etc.)
_response_cache: Dict[str, str] = {}


def _get_cache_key(prompt: str) -> str:
    """Generate cache key from prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()


def _sanitize_feature_name(name: str) -> str:
    """Convert feature name to valid Python identifier."""
    # Replace spaces and special chars with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it doesn't start with a digit
    if name and name[0].isdigit():
        name = '_' + name
    return name or 'feature'


def _create_prompt(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 10
) -> Tuple[str, List[str]]:
    """
    Create a prompt for the LLM to generate a symbolic expression.
    
    Args:
        X: Feature data (n_samples, n_features)
        y: Target values (n_samples,)
        feature_names: Names of features
        n_samples: Number of samples to include in prompt
    
    Returns:
        Tuple of (formatted prompt string, sanitized feature names)
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]
    
    # Sanitize feature names to be valid Python identifiers
    sanitized_names = [_sanitize_feature_name(name) for name in feature_names]
    
    # Sample a subset of data points for the prompt
    n_show = min(n_samples, len(X))
    indices = np.linspace(0, len(X) - 1, n_show, dtype=int)
    X_sample = X[indices]
    y_sample = y[indices]
    
    # Create feature description
    feature_desc = "\n".join([
        f"- {san_name}: range [{X[:, i].min():.3f}, {X[:, i].max():.3f}], mean={X[:, i].mean():.3f}"
        for i, (san_name) in enumerate(sanitized_names)
    ])
    
    # Create example data points
    examples = []
    for idx, (x_row, y_val) in enumerate(zip(X_sample, y_sample)):
        feature_vals = ", ".join([f"{san_name}={x_row[i]:.3f}" for i, san_name in enumerate(sanitized_names)])
        examples.append(f"  Example {idx+1}: {feature_vals} → prediction={y_val:.4f}")
    
    examples_str = "\n".join(examples)
    
    prompt = f"""You are a mathematical modeling expert. Given the following features and their relationships to predictions, generate a Python-compatible mathematical expression that approximates the relationship.

Features:
{feature_desc}

Example data points:
{examples_str}

Requirements:
1. Generate a SINGLE Python expression (no function definition, no imports)
2. Use only the feature names provided: {', '.join(sanitized_names)}
3. You can use: +, -, *, /, ** (power), np.exp, np.log, np.sin, np.cos, np.abs
4. Include polynomial terms, interactions, and non-linear transformations as needed
5. The expression should be a single line that can be evaluated with: eval(expression, {{'np': np}}, feature_dict)
6. Return ONLY the expression, nothing else

Example format:
0.5 * {sanitized_names[0]}**2 + 0.3 * {sanitized_names[0]} * {sanitized_names[1]} - 0.2 * {sanitized_names[-1]}

Generate the expression:"""
    
    return prompt, sanitized_names


def _call_openai_api(prompt: str, model: str = None) -> Optional[str]:
    """Call OpenAI API to generate expression."""
    if not OPENAI_AVAILABLE:
        return None
    
    if OPENAI_API_KEY is None:
        warnings.warn("OPENAI_API_KEY not set. Set it in environment or config.py")
        return None
    
    if model is None:
        model = DEFAULT_LLM_MODEL
    
    try:
        # Support both new (v1.0+) and old (<v1.0) versions of openai package
        if hasattr(openai, 'OpenAI'):
            # New API (openai >= 1.0)
            client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=TIMEOUT_SECONDS)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a mathematical modeling expert. Generate concise, valid Python expressions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
            )
            expression = response.choices[0].message.content.strip()
        else:
            # Old API (openai < 1.0)
            openai.api_key = OPENAI_API_KEY
            openai.request_timeout = TIMEOUT_SECONDS
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a mathematical modeling expert. Generate concise, valid Python expressions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
            )
            expression = response['choices'][0]['message']['content'].strip()
        
        return expression
    except Exception as e:
        warnings.warn(f"OpenAI API call failed: {e}")
        return None


def _call_anthropic_api(prompt: str, model: str = None) -> Optional[str]:
    """Call Anthropic API to generate expression."""
    if not ANTHROPIC_AVAILABLE:
        return None
    
    if ANTHROPIC_API_KEY is None:
        warnings.warn("ANTHROPIC_API_KEY not set. Set it in environment or config.py")
        return None
    
    if model is None:
        model = DEFAULT_ANTHROPIC_MODEL
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=TIMEOUT_SECONDS)
        
        response = client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        expression = response.content[0].text.strip()
        return expression
    except Exception as e:
        warnings.warn(f"Anthropic API call failed: {e}")
        return None


def generate_llm_expression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    provider: str = 'openai',
    model: str = None,
    use_cache: bool = True
) -> Optional[str]:
    """
    Generate a symbolic expression using LLM API.
    
    Args:
        X: Feature data
        y: Target values
        feature_names: Names of features
        provider: 'openai' or 'anthropic'
        model: Model name (optional)
        use_cache: Whether to use cached responses
    
    Returns:
        Python expression string or None if generation fails
    """
    prompt, sanitized_names = _create_prompt(X, y, feature_names)
    
    # Check cache
    if use_cache and CACHE_RESPONSES:
        cache_key = _get_cache_key(prompt)
        if cache_key in _response_cache:
            return _response_cache[cache_key]
    
    # Call appropriate API
    expression = None
    if provider.lower() == 'openai':
        expression = _call_openai_api(prompt, model)
    elif provider.lower() == 'anthropic':
        expression = _call_anthropic_api(prompt, model)
    else:
        warnings.warn(f"Unknown provider: {provider}")
        return None
    
    # Clean and validate expression
    if expression:
        expression = _clean_expression(expression)
        if _validate_expression(expression, sanitized_names):
            # Cache if valid
            if use_cache and CACHE_RESPONSES:
                cache_key = _get_cache_key(prompt)
                _response_cache[cache_key] = expression
            return expression
        else:
            warnings.warn(f"Generated expression failed validation: {expression}")
    
    return None


def _clean_expression(expression: str) -> str:
    """Clean and extract expression from LLM response."""
    # Remove markdown code blocks if present
    expression = re.sub(r'```python\n?', '', expression)
    expression = re.sub(r'```\n?', '', expression)
    expression = re.sub(r'`', '', expression)
    
    # Remove common prefixes
    expression = re.sub(r'^(expression\s*=|def\s+\w+|return\s+)', '', expression, flags=re.IGNORECASE)
    
    # Remove leading/trailing whitespace and quotes
    expression = expression.strip().strip('"').strip("'")
    
    # Remove newlines
    expression = expression.replace('\n', ' ')
    
    return expression.strip()


def _validate_expression(expression: str, feature_names: List[str]) -> bool:
    """Basic validation of expression syntax."""
    try:
        # Create a safe test environment with feature names as keys
        test_values = {name: 1.0 for name in feature_names}
        test_values['np'] = np
        
        # Try to compile and evaluate
        code = compile(expression, '<string>', 'eval')
        
        # Build safe globals with numpy functions
        safe_globals = {
            "__builtins__": {},
            "np": np,
            "abs": abs,
            "exp": np.exp,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
            "where": np.where,
            "sqrt": np.sqrt,
            "tanh": np.tanh,
        }
        
        # Try evaluation with test values
        result = eval(code, safe_globals, test_values)
        
        # Check that result is numeric or array-like
        if isinstance(result, (int, float, np.number, np.ndarray)):
            # For arrays, check if they're numeric
            if isinstance(result, np.ndarray):
                return result.dtype in [np.float32, np.float64, np.int32, np.int64, float, int]
            return True
        return False
    except Exception as e:
        # More verbose debugging
        # print(f"Validation error: {e}")
        return False


def compile_expression(expression: str, feature_names: List[str]) -> callable:
    """
    Compile expression into a callable function.
    
    Args:
        expression: Python expression string (using sanitized feature names)
        feature_names: Original names of features (will be mapped to sanitized names)
    
    Returns:
        Function that takes X (n_samples, n_features) and returns predictions
    """
    # Create mapping from original to sanitized names
    sanitized_names = [_sanitize_feature_name(name) for name in feature_names]
    
    # Create namespace with numpy
    namespace = {'np': np}
    
    # Compile expression
    code = compile(expression, '<string>', 'eval')
    
    def predict_fn(X: np.ndarray) -> np.ndarray:
        """Predict using compiled expression."""
        predictions = []
        for x_row in X:
            # Create feature dictionary with sanitized names
            feature_dict = {san_name: x_row[i] for i, san_name in enumerate(sanitized_names)}
            feature_dict.update(namespace)
            
            try:
                pred = eval(code, {"__builtins__": {}, "np": np, "abs": abs, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos}, feature_dict)
                predictions.append(float(pred))
            except Exception as e:
                warnings.warn(f"Expression evaluation failed: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    return predict_fn
