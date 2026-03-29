"""
Gradio UI components for the Classification Tab.
"""
import gradio as gr
import pandas as pd
from syllabusml.classification.engine import ClassificationEngine
from syllabusml.config import get_config

def create_classification_tab():
    engine = ClassificationEngine()
    cfg = get_config()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Dataset Setup")
            file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
            load_iris_btn = gr.Button("Load Iris (Classification)", variant="secondary")
            
            with gr.Accordion("Features & Target", open=True):
                target_col = gr.Dropdown(label="Select Target Variable")
                feature_cols = gr.CheckboxGroup(label="Select Features")
            
            with gr.Tabs():
                with gr.TabItem("Individual Lab"):
                    algo_select = gr.Radio(
                        ["Decision Tree", "KNN", "Logistic Regression"], 
                        label="Select Algorithm", value="Decision Tree"
                    )
                    # Dynamic params
                    dt_depth = gr.Slider(1, 20, value=5, label="Max Depth (DT)", visible=True)
                    knn_n = gr.Slider(1, 15, value=5, label="Neighbors (KNN)", visible=False)
                    
                    train_btn = gr.Button("🚀 Train Classifier", variant="primary")
                
                with gr.TabItem("Mini Project (Lab 10)"):
                    gr.Markdown("Compare 4 classifiers automatically.")
                    compare_btn = gr.Button("📊 Run Comparison Report", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Results & Confusion Matrix")
            metrics_view = gr.JSON(label="Performance Metrics")
            cm_plot = gr.Plot(label="Confusion Matrix")
            comparison_plot = gr.Plot(label="Comparison Plot")
            comparison_table = gr.DataFrame(label="Comparison Table")

    # Visibility Toggle Logic
    def toggle_params(algo):
        return gr.update(visible=algo == "Decision Tree"), gr.update(visible=algo == "KNN")

    algo_select.change(toggle_params, inputs=[algo_select], outputs=[dt_depth, knn_n])

    # Loading Logic
    def handle_file_upload(file):
        if file is None: return None, gr.update(choices=[]), gr.update(choices=[])
        df = pd.read_csv(file.name)
        cols = df.columns.tolist()
        return df, gr.update(choices=cols, value=cols[-1]), gr.update(choices=cols, value=cols[:-1])

    def load_data():
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        df = pd.read_csv(url)
        cols = df.columns.tolist()
        return df, gr.update(choices=cols, value="species"), gr.update(choices=cols[:-1], value=cols[:-1])

    df_state = gr.State()
    file_input.change(handle_file_upload, inputs=[file_input], outputs=[df_state, target_col, feature_cols])
    load_iris_btn.click(load_data, outputs=[df_state, target_col, feature_cols])

    # Training Logic
    def run_individual(df, target, features, algo, depth, n):
        if df is None: return gr.Error("Please load a dataset first."), None, gr.update(visible=False), gr.update(visible=False)
        if not features: return gr.Error("Please select at least one feature."), None, gr.update(visible=False), gr.update(visible=False)
        if not target: return gr.Error("Please select a target variable."), None, gr.update(visible=False), gr.update(visible=False)

        params = {}
        if algo == "Decision Tree": params = {"max_depth": depth}
        elif algo == "KNN": params = {"n_neighbors": n}
        
        try:
            metrics, cm, classes = engine.train_and_evaluate(df, features, target, algo, params)
            fig = engine.plot_confusion_matrix(cm, classes)
            return metrics, fig, gr.update(visible=False), gr.update(visible=False)
        except Exception as e:
            return gr.Error(f"Training Error: {str(e)}"), None, gr.update(visible=False), gr.update(visible=False)

    # Comparison Logic (Lab 10)
    def run_comparison(df, target, features):
        if df is None or not features or not target:
            return gr.Error("Missing data/selections"), None, gr.update(visible=False), gr.update(visible=False)
        
        try:
            results_df = engine.compare_models(df, features, target)
            fig = engine.plot_comparison(results_df)
            return None, None, fig, results_df
        except Exception as e:
            return gr.Error(f"Comparison Error: {str(e)}"), None, gr.update(visible=False), gr.update(visible=False)

    train_btn.click(
        run_individual, 
        inputs=[df_state, target_col, feature_cols, algo_select, dt_depth, knn_n],
        outputs=[metrics_view, cm_plot, comparison_plot, comparison_table]
    )

    compare_btn.click(
        run_comparison,
        inputs=[df_state, target_col, feature_cols],
        outputs=[metrics_view, cm_plot, comparison_plot, comparison_table]
    )
