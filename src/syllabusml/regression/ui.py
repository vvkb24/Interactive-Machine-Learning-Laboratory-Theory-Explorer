"""
Gradio UI components for the Regression Tab.
"""
import gradio as gr
import pandas as pd
from syllabusml.regression.engine import RegressionEngine
from syllabusml.config import get_config
import os

def create_regression_tab():
    """Constructs the Regression layout."""
    engine = RegressionEngine()
    cfg = get_config()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Dataset Setup")
            file_input = gr.File(label="Upload Dataset", file_types=[".csv"])
            load_house_btn = gr.Button("Load Boston/California Housing", variant="secondary")
            
            gr.Markdown("### 🎯 Model Config")
            target_col = gr.Dropdown(label="Select Target Variable (Y)")
            feature_cols = gr.CheckboxGroup(label="Select Features (X)")
            
            with gr.Row():
                test_size = gr.Slider(0.1, 0.4, value=cfg.model.test_size, label="Test Size")
                random_state = gr.Number(value=cfg.model.random_state, label="Random State")
            
            train_btn = gr.Button("🚀 Train Regression Model", variant="primary")
            save_btn = gr.Button("💾 Save Model")

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Performance Analysis")
            metrics_view = gr.JSON(label="Regression Metrics (R2, MSE, MAE)")
            reg_plot = gr.Plot(label="Actual vs Predicted")
            
            gr.Markdown("### 🧪 Model Coefficients")
            coef_table = gr.DataFrame(label="Feature Importances (Coefficients)")

    # Data Loading Logic
    def load_data(file):
        if file is None: return None, gr.update(choices=[]), gr.update(choices=[])
        df = pd.read_csv(file.name)
        cols = df.select_dtypes(include=['number']).columns.tolist()
        return df, gr.update(choices=cols), gr.update(choices=cols)

    def load_sample_housing():
        # Load sample housing data for Lab 5
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        df = pd.read_csv(url)
        cols = df.columns.tolist()
        return df, gr.update(choices=cols, value="medv"), gr.update(choices=cols, value=cols[:-1])

    # Training Logic
    def run_training(df, target, features, split, seed):
        if not features or not target:
            return {"Error": "Please select features and target"}, None, None
        
        metrics, y_test, y_pred = engine.train_model(df, features, target, split, int(seed))
        fig = engine.plot_regression_results(y_test, y_pred)
        
        # Prepare coefficient df
        coeffs = pd.DataFrame({
            "Feature": list(metrics["Coefficients"].keys()) + ["Intercept"],
            "Weight": list(metrics["Coefficients"].values()) + [metrics["Intercept"]]
        })
        
        # Remove raw weights from main metrics JSON for clean view
        display_metrics = {k: v for k, v in metrics.items() if k not in ["Coefficients", "Intercept"]}
        
        return display_metrics, fig, coeffs

    # Events
    file_input.change(load_data, inputs=[file_input], outputs=[gr.State(), target_col, feature_cols])
    load_house_btn.click(load_sample_housing, outputs=[gr.State(), target_col, feature_cols])
    
    # We use a State variable for the dataframe to keep it available between interactions
    df_state = gr.State()
    
    # Internal update of df_state
    file_input.change(lambda f: pd.read_csv(f.name) if f else None, inputs=[file_input], outputs=[df_state])
    load_house_btn.click(lambda: pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"), outputs=[df_state])

    train_btn.click(
        run_training, 
        inputs=[df_state, target_col, feature_cols, test_size, random_state], 
        outputs=[metrics_view, reg_plot, coef_table]
    )
    
    save_btn.click(lambda: engine.save_model(), outputs=[gr.Textbox(label="Status")])
