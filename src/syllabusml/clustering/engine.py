"""
Core logic for Clustering algorithms.
Implements Lab 9: K-Means Clustering, Elbow Method, and Silhouette Analysis.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

class ClusteringEngine:
    """
    Engine for unsupervised learning using K-Means.
    Includes diagnostic tools like Elbow Method and Silhouette Analysis.
    """
    
    def __init__(self):
        self.last_df = None
        self.last_features = None

    def run_kmeans(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        n_clusters: int = 3,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, float]:
        """
        Trains K-Means and returns the clustered dataframe + silhouette score.
        
        Why: K-Means is the standard unsupervised algorithm for Lab 9.
        Failure Mode: Requires numeric features; fails if n_clusters > n_samples.
        """
        X = df[features].copy()
        
        # Financial datasets often have commas or mixed types in numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Drop rows with NaN that couldn't be coerced (essential for Clustering to avoid length mismatch)
        X = X.dropna()
        
        # Only keep rows in the original dataframe that survived numeric coercion to avoid shape mismatch
        df_cleaned = df.loc[X.index].copy()
        
        # Standardize for better clustering if features have different scales
        X_norm = (X - X.mean()) / X.std()
        X_norm = X_norm.fillna(0) # Handle constant columns
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        df_cleaned['Cluster'] = kmeans.fit_predict(X_norm)
        
        score = silhouette_score(X_norm, df_cleaned['Cluster'])
        return df_cleaned, score

    def compute_elbow(self, df: pd.DataFrame, features: List[str], max_k: int = 10) -> Any:
        """
        Computes the Within-Cluster Sum of Squares (WCSS) for the Elbow Method.
        """
        X = df[features].copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col].astype(str).str.replace(',', ''), errors='coerce')
        
        X = X.dropna()
        X_norm = (X - X.mean()) / X.std()
        X_norm = X_norm.fillna(0)
        
        wcss = []
        k_range = range(1, min(max_k, len(X)) + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X_norm)
            wcss.append(kmeans.inertia_)
            
        fig = px.line(
            x=list(k_range), y=wcss, 
            markers=True,
            title="Elbow Method (WCSS vs K)",
            labels={'x': 'Number of Clusters (k)', 'y': 'WCSS'},
            template="plotly_white"
        )
        return fig

    def plot_clusters(self, df: pd.DataFrame, x_feat: str, y_feat: str) -> Any:
        """Generates a 2D scatter plot of the clusters."""
        fig = px.scatter(
            df, x=x_feat, y=y_feat, color='Cluster',
            title=f"K-Means Clusters: {x_feat} vs {y_feat}",
            template="plotly_white",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(coloraxis_showscale=False)
        return fig
