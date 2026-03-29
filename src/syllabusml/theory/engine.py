"""
Core logic for Theory Visualizations.
Implements interactive demonstrations for Perceptrons, PCA, and Neural Network intuition.
Uses Plotly for animations and mathematical visuals.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from typing import Any, Tuple

class TheoryEngine:
    """
    Engine for generating educational visuals and animations for ML theory.
    """

    @staticmethod
    def simulate_perceptron(w1: float, w2: float, b: float) -> Any:
        """
        Visualizes the Linear Separability of a Perceptron in 2D.
        
        Formula: y = step(w1*x1 + w2*x2 + b)
        Decision Boundary: x2 = -(w1/w2)x1 - (b/w2)
        """
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X1, X2 = np.meshgrid(x, y)
        Z = w1 * X1 + w2 * X2 + b
        Z_step = np.where(Z >= 0, 1, 0)

        fig = go.Figure(data=go.Contour(
            x=x, y=y, z=Z_step,
            colorscale='RdBu',
            opacity=0.6,
            showscale=False,
            contours_coloring='heatmap'
        ))

        # Add the decision boundary line
        if w2 != 0:
            y_bound = -(w1 / w2) * x - (b / w2)
            fig.add_trace(go.Scatter(x=x, y=y_bound, mode='lines', name='Decision Boundary', 
                                   line=dict(color='black', width=3)))
        
        fig.update_layout(title="Perceptron Decision Boundary (Linear Separability)",
                          xaxis_title="Feature 1 (x1)", yaxis_title="Feature 2 (x2)",
                          xaxis=dict(range=[-10, 10]), yaxis=dict(range=[-10, 10]),
                          template="plotly_white")
        return fig

    @staticmethod
    def simulate_pca_projection(n_points: int = 200, angle: float = 45) -> Any:
        """
        Demonstrates Dimensionality Reduction (PCA) by projecting 2D data onto 1D.
        """
        # Generate correlated 2D data
        np.random.seed(42)
        x = np.random.normal(0, 1, n_points)
        y = x * np.tan(np.radians(angle)) + np.random.normal(0, 0.2, n_points)
        data = np.vstack([x, y]).T

        pca = PCA(n_components=1)
        data_pca = pca.fit_transform(data)
        data_projected = pca.inverse_transform(data_pca)

        fig = go.Figure()
        # Original points
        fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers', name='Original Data',
                               marker=dict(color='blue', opacity=0.3)))
        # Principal Component Axis
        fig.add_trace(go.Scatter(x=data_projected[:, 0], y=data_projected[:, 1], mode='markers', 
                               name='PCA Projection (1D)', marker=dict(color='red', size=4)))
        
        fig.update_layout(title=f"PCA: Dimensionality Reduction (Variance Explained: {pca.explained_variance_ratio_[0]:.2f})",
                          xaxis_title="Feature 1", yaxis_title="Feature 2",
                          template="plotly_white")
        return fig

    @staticmethod
    def get_theory_explanation(topic: str) -> str:
        """
        Retrieves the rich theory explanation for a specific ML concept.
        
        Args:
            topic (str): The name of the ML topic (e.g., 'Perceptron').
            
        Returns:
            str: Rich Markdown formatted explanation.
        """
        from . import explanations
        
        mapping = {
            "Perceptron & Linear Separability": explanations.get_perceptron_explanation,
            "MLP & Backpropagation": explanations.get_mlp_explanation,
            "Ensemble Methods (Intuition)": explanations.get_trees_explanation,
            "Support Vector Machines (SVM)": explanations.get_svm_explanation,
            "Dimensionality Reduction (PCA)": explanations.get_pca_explanation,
            "Genetic Algorithms": explanations.get_ga_explanation,
            "Reinforcement Learning": explanations.get_rl_explanation,
        }
        
        func = mapping.get(topic, None)
        if func:
            return func()
        
        return "### Concept Not Found\nSelect a valid topic from the menu to see the explanation."

    @staticmethod
    def get_neural_net_markdown() -> str:
        """Returns mathematical intuition for Backpropagation."""
        return r"""
        ### 🧠 Multi-Layer Perceptron (MLP) & Backpropagation
        
        **1. Forward Pass:**
        The input $x$ is multiplied by weights $W$, biases $b$ are added, and an activation function $\sigma$ (like ReLU or Sigmoid) is applied:
        $$a = \sigma(Wx + b)$$
        
        **2. Loss Computation:**
        The error is calculated using a Cost Function (e.g., Mean Squared Error):
        $$J = \frac{1}{2}(y - \hat{y})^2$$
        
        **3. Backpropagation (The Chain Rule):**
        To update weights, we find the gradient of the loss with respect to each weight using the chain rule:
        $$\frac{\partial J}{\partial W} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}$$
        
        **4. Optimization:**
        Update weights using Gradient Descent:
        $$W_{new} = W_{old} - \eta \cdot \nabla J(W)$$
        """

    @staticmethod
    def get_theory_explanation(topic: str) -> str:
        """
        Retrieves the rich theory explanation for a specific ML concept.
        
        Args:
            topic (str): The name of the ML topic (e.g., 'Perceptron').
            
        Returns:
            str: Rich Markdown formatted explanation.
        """
        from syllabusml.theory import explanations
        
        mapping = {
            "Perceptron & Linear Separability": explanations.get_perceptron_explanation,
            "Dimensionality Reduction (PCA)": explanations.get_dr_explanation,
            "MLP & Backpropagation": explanations.get_mlp_explanation,
            "Ensemble Methods (Intuition)": explanations.get_trees_explanation,
            "Support Vector Machines (SVM)": explanations.get_svm_explanation,
            "Genetic Algorithms": explanations.get_ga_explanation,
            "Reinforcement Learning": explanations.get_rl_explanation,
        }
        
        func = mapping.get(topic, None)
        if func:
            return func()
        
        return "### Concept Not Found\nSelect a valid topic from the menu to see the explanation."
