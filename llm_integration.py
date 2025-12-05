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


def _create_prompt(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 10
) -> str:
    """
    Create a prompt for the LLM to generate a symbolic expression.
    
    Args:
        X: Feature data (n_samples, n_features)
        y: Target values (n_samples,)
        feature_names: Names of features
        n_samples: Number of samples to include in prompt
    
    Returns:
        Formatted prompt string
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]
    
    # Sample a subset of data points for the prompt
    n_show = min(n_samples, len(X))
    indices = np.linspace(0, len(X) - 1, n_show, dtype=int)
    X_sample = X[indices]
    y_sample = y[indices]
    
    # Create feature description
    feature_desc = "\n".join([
        f"- {name}: range [{X[:, i].min():.3f}, {X[:, i].max():.3f}], mean={X[:, i].mean():.3f}"
        for i, name in enumerate(feature_names)
    ])
    
    # Create example data points
    examples = []
    for idx, (x_row, y_val) in enumerate(zip(X_sample, y_sample)):
        feature_vals = ", ".join([f"{name}={x_row[i]:.3f}" for i, name in enumerate(feature_names)])
        examples.append(f"  Example {idx+1}: {feature_vals} → prediction={y_val:.4f}")
    
    examples_str = "\n".join(examples)
    
    prompt = f"""You are a mathematical modeling expert. Given the following features and their relationships to predictions, generate a Python-compatible mathematical expression that approximates the relationship.

Features:
{feature_desc}

Example data points:
{examples_str}

Requirements:
1. Generate a SINGLE Python expression (no function definition, no imports)
2. Use only the feature names provided: {', '.join(feature_names)}
3. You can use: +, -, *, /, ** (power), np.exp, np.log, np.sin, np.cos, np.abs
4. Include polynomial terms, interactions, and non-linear transformations as needed
5. The expression should be a single line that can be evaluated with: eval(expression, {{'np': np}}, feature_dict)
6. Return ONLY the expression, nothing else

Example format:
0.5 * {feature_names[0]}**2 + 0.3 * {feature_names[0]} * {feature_names[1]} - 0.2 * {feature_names[-1]}

Generate the expression:"""
    
    return prompt


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
        client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=TIMEOUT_SECONDS)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a mathematical modeling expert. Generate concise, valid Python expressions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more deterministic outputs
            max_tokens=200,
        )
        
        expression = response.choices[0].message.content.strip()
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
    prompt = _create_prompt(X, y, feature_names)
    
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
        if _validate_expression(expression, feature_names or [f"x_{i}" for i in range(X.shape[1])]):
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
        # Create a safe test environment
        test_values = {name: 1.0 for name in feature_names}
        test_values['np'] = np
        
        # Try to evaluate (with limited scope)
        allowed_names = set(feature_names) | {'np'}
        code = compile(expression, '<string>', 'eval')
        
        # Check that only allowed names are used
        used_names = set(code.co_names)
        if not used_names.issubset(allowed_names | {'abs', 'exp', 'log', 'sin', 'cos'}):
            return False
        
        # Try evaluation
        eval(code, {"__builtins__": {}}, test_values)
        return True
    except Exception:
        return False


def compile_expression(expression: str, feature_names: List[str]) -> callable:
    """
    Compile expression into a callable function.
    
    Args:
        expression: Python expression string
        feature_names: Names of features
    
    Returns:
        Function that takes X (n_samples, n_features) and returns predictions
    """
    # Create namespace with numpy
    namespace = {'np': np}
    
    # Compile expression
    code = compile(expression, '<string>', 'eval')
    
    def predict_fn(X: np.ndarray) -> np.ndarray:
        """Predict using compiled expression."""
        predictions = []
        for x_row in X:
            # Create feature dictionary
            feature_dict = {name: x_row[i] for i, name in enumerate(feature_names)}
            feature_dict.update(namespace)
            
            try:
                pred = eval(code, {"__builtins__": {}}, feature_dict)
                predictions.append(float(pred))
            except Exception as e:
                warnings.warn(f"Expression evaluation failed: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    return predict_fn

