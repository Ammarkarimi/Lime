"""
LLM-Enhanced LIME: Non-linear model wrapper using LLMs for better local approximations.

This module provides a wrapper that uses LLMs to generate non-linear models
on-the-fly for LIME explanations, improving accuracy when linear models fail.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
import warnings


class LLMNonLinearRegressor(BaseEstimator, RegressorMixin):
    """
    A non-linear regressor that uses LLM-generated models and extracts
    feature importance compatible with LIME's linear model interface.
    
    This class wraps a non-linear model (generated via LLM or other means)
    and provides a sklearn-compatible interface with feature importance
    extraction using gradient-based or permutation-based methods.
    """
    
    def __init__(
        self,
        model_fn: Optional[Callable] = None,
        feature_names: Optional[List[str]] = None,
        importance_method: str = 'gradient',
        llm_provider: str = 'openai',
        llm_model: str = 'gpt-4',
        use_simple_polynomial: bool = True,
        max_degree: int = 3,
        random_state: Optional[int] = None
    ):
        """
        Initialize the LLM-based non-linear regressor.
        
        Args:
            model_fn: Optional pre-defined model function. If None, will generate
                     using LLM or polynomial approximation.
            feature_names: Names of features for better LLM prompts.
            importance_method: Method to extract feature importance.
                             Options: 'gradient', 'permutation', 'shap'
            llm_provider: LLM provider ('openai', 'anthropic', 'local', 'none')
            llm_model: Model name for LLM provider
            use_simple_polynomial: If True, use polynomial features as fallback
                                  when LLM is not available
            max_degree: Maximum degree for polynomial features
            random_state: Random seed for reproducibility
        """
        self.model_fn = model_fn
        self.feature_names = feature_names
        self.importance_method = importance_method
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.use_simple_polynomial = use_simple_polynomial
        self.max_degree = max_degree
        self.random_state = random_state
        
        # Will be set during fit
        self.n_features_ = None
        self.coef_ = None
        self.intercept_ = None
        self.fitted_model_ = None
        self.feature_importance_ = None
        
    def _generate_llm_model(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: Optional[List[str]] = None) -> Callable:
        """
        Generate a non-linear model using LLM or fallback methods.
        
        Tries to use LLM to generate symbolic expression, falls back to
        polynomial features or neural network if LLM fails.
        """
        # Try LLM first if provider is set
        if self.llm_provider.lower() in ['openai', 'anthropic']:
            try:
                from .llm_integration import generate_llm_expression, compile_expression
                
                if feature_names is None:
                    feature_names = [f"x_{i}" for i in range(X.shape[1])]
                
                expression = generate_llm_expression(
                    X, y, feature_names, 
                    provider=self.llm_provider,
                    model=self.llm_model
                )
                
                if expression:
                    # Compile expression into function
                    model_fn = compile_expression(expression, feature_names)
                    
                    # Wrap in a class for sklearn compatibility
                    class ExpressionModel:
                        def __init__(self, predict_fn):
                            self.predict_fn = predict_fn
                        
                        def fit(self, X, y, sample_weight=None):
                            # Expression models don't need fitting
                            return self
                        
                        def predict(self, X):
                            return self.predict_fn(X)
                    
                    return ExpressionModel(model_fn)
            except Exception as e:
                warnings.warn(f"LLM model generation failed: {e}. Falling back to polynomial features.")
        
        # Fallback to polynomial features or neural network
        if self.use_simple_polynomial:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import Pipeline
            
            poly = PolynomialFeatures(degree=min(self.max_degree, 3), include_bias=False)
            ridge = Ridge(alpha=0.1, random_state=self.random_state)
            model = Pipeline([('poly', poly), ('ridge', ridge)])
            return model
        else:
            # Use neural network
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(
                hidden_layer_sizes=(50, 25),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            )
            return model
    
    def _extract_feature_importance_gradient(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract feature importance using gradient-based method.
        
        Computes the gradient of the model output w.r.t. each feature
        at the data points, weighted by sample weights.
        """
        try:
            from scipy.optimize import approx_fprime
        except ImportError:
            warnings.warn("scipy not available, falling back to permutation importance")
            return self._extract_feature_importance_permutation(X, y, sample_weight)
        
        # Compute gradients at each point
        gradients = []
        for i in range(X.shape[0]):
            x = X[i]
            # Numerical gradient
            eps = 1e-5
            grad = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                x_plus = x.copy()
                x_plus[j] += eps
                x_minus = x.copy()
                x_minus[j] -= eps
                
                pred_plus = self.fitted_model_.predict(x_plus.reshape(1, -1))[0]
                pred_minus = self.fitted_model_.predict(x_minus.reshape(1, -1))[0]
                grad[j] = (pred_plus - pred_minus) / (2 * eps)
            
            gradients.append(grad)
        
        gradients = np.array(gradients)
        
        # Weight by sample weights and average
        if sample_weight is not None:
            weights = sample_weight / sample_weight.sum()
            importance = np.average(np.abs(gradients), axis=0, weights=weights)
        else:
            importance = np.mean(np.abs(gradients), axis=0)
        
        return importance
    
    def _extract_feature_importance_permutation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract feature importance using permutation importance.
        
        Measures how much the model's prediction changes when a feature is shuffled.
        """
        baseline_pred = self.fitted_model_.predict(X)
        if sample_weight is not None:
            baseline_score = np.average((baseline_pred - y) ** 2, weights=sample_weight)
        else:
            baseline_score = np.mean((baseline_pred - y) ** 2)
        
        importance = np.zeros(X.shape[1])
        rng = np.random.RandomState(self.random_state)
        
        for j in range(X.shape[1]):
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, j])
            permuted_pred = self.fitted_model_.predict(X_permuted)
            
            if sample_weight is not None:
                permuted_score = np.average((permuted_pred - y) ** 2, weights=sample_weight)
            else:
                permuted_score = np.mean((permuted_pred - y) ** 2)
            
            # Importance is the increase in error
            importance[j] = permuted_score - baseline_score
        
        # Normalize to make it comparable to linear coefficients
        importance = importance / (np.abs(importance).sum() + 1e-10)
        
        return importance
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Fit the non-linear model and extract feature importance.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
            sample_weight: Sample weights (n_samples,)
        """
        self.n_features_ = X.shape[1]
        
        # Generate or use provided model
        if self.model_fn is None:
            self.fitted_model_ = self._generate_llm_model(X, y, self.feature_names)
        else:
            self.fitted_model_ = self.model_fn
        
        # Fit the model
        if hasattr(self.fitted_model_, 'fit'):
            if sample_weight is not None:
                # If fitted_model_ is a Pipeline, forward sample_weight to the final step
                try:
                    if isinstance(self.fitted_model_, Pipeline):
                        last_step_name = self.fitted_model_.steps[-1][0]
                        params = {f"{last_step_name}__sample_weight": sample_weight}
                        self.fitted_model_.fit(X, y, **params)
                    else:
                        # Try to pass sample_weight directly
                        self.fitted_model_.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    # Some models declare fit but don't accept sample_weight
                    self.fitted_model_.fit(X, y)
                except Exception:
                    # Fallback to fitting without sample_weight on any other exception
                    try:
                        self.fitted_model_.fit(X, y)
                    except Exception:
                        # Re-raise original exception if fit without sample_weight also fails
                        raise
            else:
                self.fitted_model_.fit(X, y)
        
        # Extract feature importance
        if self.importance_method == 'gradient':
            self.feature_importance_ = self._extract_feature_importance_gradient(
                X, y, sample_weight
            )
        elif self.importance_method == 'permutation':
            self.feature_importance_ = self._extract_feature_importance_permutation(
                X, y, sample_weight
            )
        else:
            raise ValueError(f"Unknown importance method: {self.importance_method}")
        
        # Convert to linear model format (coef_ and intercept_)
        # For LIME compatibility, we approximate the non-linear model
        # with a linear one using the extracted importance
        self.coef_ = self.feature_importance_
        
        # Compute intercept as the mean prediction at the mean of features
        X_mean = X.mean(axis=0).reshape(1, -1)
        mean_pred = self.predict(X_mean)[0]
        X_mean_weighted = np.average(X, axis=0, weights=sample_weight) if sample_weight is not None else X.mean(axis=0)
        intercept_pred = mean_pred - np.dot(self.coef_, X_mean_weighted)
        self.intercept_ = intercept_pred
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted non-linear model."""
        if self.fitted_model_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.fitted_model_.predict(X)
    
    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Compute R² score."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


class LLMEnhancedLimeExplainer:
    """
    Enhanced LIME explainer that uses LLM-generated non-linear models
    for better local approximations.
    """
    
    def __init__(
        self,
        base_explainer,
        use_llm_model: bool = True,
        importance_method: str = 'gradient',
        llm_provider: str = 'openai',
        **llm_kwargs
    ):
        """
        Initialize the LLM-enhanced LIME explainer.
        
        Args:
            base_explainer: Base LIME explainer (e.g., LimeTabularExplainer)
            use_llm_model: Whether to use LLM-based non-linear models
            importance_method: Method for extracting feature importance
            llm_provider: LLM provider to use
            **llm_kwargs: Additional arguments for LLMNonLinearRegressor
        """
        self.base_explainer = base_explainer
        self.use_llm_model = use_llm_model
        self.importance_method = importance_method
        self.llm_provider = llm_provider
        self.llm_kwargs = llm_kwargs
    
    def explain_instance(
        self,
        data_row,
        predict_fn,
        labels=(1,),
        top_labels=None,
        num_features=10,
        num_samples=5000,
        **kwargs
    ):
        """
        Explain an instance using LLM-enhanced LIME.
        
        Creates a non-linear model regressor and passes it to the base explainer.
        """
        if self.use_llm_model:
            # Get feature names if available
            feature_names = getattr(self.base_explainer, 'feature_names', None)
            
            # Create LLM-based regressor
            llm_regressor = LLMNonLinearRegressor(
                feature_names=feature_names,
                importance_method=self.importance_method,
                llm_provider=self.llm_provider,
                **self.llm_kwargs
            )
            
            # Pass to base explainer
            kwargs['model_regressor'] = llm_regressor
        
        # Call base explainer
        return self.base_explainer.explain_instance(
            data_row=data_row,
            predict_fn=predict_fn,
            labels=labels,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=num_samples,
            **kwargs
        )
