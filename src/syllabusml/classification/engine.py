"""
Core logic for Classification algorithms.
Implements Lab 6 (Decision Tree), Lab 7 (KNN), Lab 8 (Logistic Regression).
Implements Lab 10 (Classifier Performance Comparison Mini-Project).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

class ClassificationEngine:
    """
    Engine for training, evaluating and comparing classification models.
    """
    
    def __init__(self):
        self.models = {
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier()
        }

    def train_and_evaluate(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        target: str, 
        algorithm: str,
        params: Dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Trains a specific classifier and returns metrics and plots.
        """
        if df is None:
            raise ValueError("No dataset provided. Please upload or load a dataset first.")
        if not features:
            raise ValueError("No features selected. Please select at least one feature.")
        if not target:
            raise ValueError("No target variable selected.")

        X = df[features].copy()
        y = df[target].copy()
        
        # Financial datasets often have commas or mixed types in numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Fill missing values created by coercion
        X = X.fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Initialize model with user params
        if algorithm == "Decision Tree":
            model = DecisionTreeClassifier(**params)
        elif algorithm == "KNN":
            model = KNeighborsClassifier(**params)
        elif algorithm == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, **params)
        else:
            model = RandomForestClassifier(**params)
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        }
        
        cm = confusion_matrix(y_test, y_pred)
        return metrics, cm, model.classes_

    def compare_models(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Mini Project (Lab 10): Compares 4 classifiers and returns a summary table.
        """
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        results = []
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({
                "Algorithm": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1 Score": f1_score(y_test, y_pred, average='weighted')
            })
            
        return pd.DataFrame(results)

    def plot_confusion_matrix(self, cm: np.ndarray, classes: np.ndarray) -> Any:
        """Generates an interactive Heatmap for the Confusion Matrix."""
        fig = px.imshow(
            cm,
            text_auto=True,
            x=classes.astype(str),
            y=classes.astype(str),
            labels=dict(x="Predicted", y="Actual", color="Count"),
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        return fig

    def plot_comparison(self, results_df: pd.DataFrame) -> Any:
        """Generates a bar chart comparing accuracies across models."""
        fig = px.bar(
            results_df, 
            x="Algorithm", 
            y="Accuracy", 
            color="Algorithm",
            title="Algorithm Performance Comparison (Lab 10)",
            text_auto='.3f'
        )
        return fig
