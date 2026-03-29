"""
Core logic for Regression models.
Implements Lab 4 (Simple Linear Regression) and Lab 5 (Multiple Linear Regression).
Uses Scikit-Learn for model implementation and metrics.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, Tuple, List
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

class RegressionEngine:
    """
    Engine for training and evaluating linear regression models.
    Supports Simple and Multiple Linear Regression.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None
        self.target_name = None

    def train_model(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        target: str, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Trains a Linear Regression model and returns performance metrics.
        
        Why: Standard Scikit-Learn implementation is production-grade and highly optimized.
        Failure Mode: If features or target are missing/non-numeric.
        """
        X = df[features]
        y = df[target]
        
        self.feature_names = features
        self.target_name = target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "R2 Score": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "Coefficients": dict(zip(features, self.model.coef_)),
            "Intercept": float(self.model.intercept_)
        }
        
        return metrics, y_test, y_pred

    def plot_regression_results(self, y_test: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        Generates an Actual vs Predicted plot.
        """
        fig = px.scatter(
            x=y_test, y=y_pred, 
            labels={'x': 'Actual Value', 'y': 'Predicted Value'},
            title="Comparison: Actual vs Predicted",
            template="plotly_white",
            trendline="ols",
            trendline_color_override="red"
        )
        # Add equality line
        fig.add_shape(
            type="line", line=dict(dash="dash"),
            x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max()
        )
        return fig

    def save_model(self, path: str = "artifacts/regression_model.joblib"):
        """Saves the trained model using joblib."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "target_name": self.target_name
        }, save_path)
        return str(save_path)
