"""
Core logic for Descriptive Statistics and Exploratory Data Analysis.
Implements Lab 1, 2, and 3: Central Tendency, Dispersion, and Visualization.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go

class StatisticsEngine:
    """
    Engine for computing statistical metrics and generating EDA plots.
    Designed for scalability using vectorized NumPy/Pandas operations.
    """
    
    @staticmethod
    def compute_central_tendency(df: pd.DataFrame, column: str) -> Dict[str, float]:
        """
        Computes Mean, Median, and Mode for a numeric column.
        
        Why: Vectorized pandas operations are O(n) and memory-efficient.
        Failure Mode: Returns None if column is non-numeric or empty.
        """
        series = df[column].dropna()
        if series.empty or not np.issubdtype(series.dtype, np.number):
            return {"Mean": 0.0, "Median": 0.0, "Mode": 0.0}
            
        return {
            "Mean": float(series.mean()),
            "Median": float(series.median()),
            "Mode": float(series.mode().iloc[0]) if not series.mode().empty else 0.0
        }

    @staticmethod
    def compute_dispersion(df: pd.DataFrame, column: str) -> Dict[str, float]:
        """
        Computes Variance and Standard Deviation.
        """
        series = df[column].dropna()
        if series.empty or not np.issubdtype(series.dtype, np.number):
            return {"Variance": 0.0, "Std Dev": 0.0, "Range": 0.0}
            
        return {
            "Variance": float(series.var()),
            "Std Dev": float(series.std()),
            "Range": float(series.max() - series.min())
        }

    @staticmethod
    def generate_eda_plots(df: pd.DataFrame, column: str) -> Tuple[Any, Any]:
        """
        Generates interactive Histogram and Box plots using Plotly.
        
        Why: Plotly provides client-side interactivity (zoom/hover) which is 
        superior to static Matplotlib for educational EDA.
        """
        if column not in df.columns:
            return None, None
            
        # Histogram with KDE-like distribution
        fig_hist = px.histogram(
            df, x=column, marginal="rug", 
            title=f"Distribution of {column}",
            template="plotly_white",
            color_discrete_sequence=["#636EFA"]
        )
        
        # Box plot for outlier detection
        fig_box = px.box(
            df, y=column, 
            title=f"Outlier Analysis for {column}",
            template="plotly_white",
            color_discrete_sequence=["#EF553B"]
        )
        
        return fig_hist, fig_box

    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame) -> Any:
        """
        Generates a heatmap of the correlation matrix for all numeric columns.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
            
        corr = numeric_df.corr()
        fig = px.imshow(
            corr, text_auto=True, 
            aspect="auto", 
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r"
        )
        return fig
