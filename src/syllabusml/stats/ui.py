"""
Gradio UI components for the Statistics & EDA Tab.
"""
import gradio as gr
import pandas as pd
from syllabusml.stats.engine import StatisticsEngine
from syllabusml.utils import load_dataset
import os

def create_stats_tab():
    """
    Constructs the Statistics & EDA layout.
    """
    engine = StatisticsEngine()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Data Input")
            file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
            load_sample_btn = gr.Button("Load Sample (Iris)", variant="secondary")
            
            gr.Markdown("### ⚙️ Parameters")
            column_selector = gr.Dropdown(label="Select Column for Analysis", choices=[])
            run_analysis_btn = gr.Button("🚀 Run Analysis", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📈 Visual Analysis")
            with gr.Tabs():
                with gr.TabItem("Distribution"):
                    hist_plot = gr.Plot(label="Histogram")
                with gr.TabItem("Outliers"):
                    box_plot = gr.Plot(label="Box Plot")
                with gr.TabItem("Correlations"):
                    corr_plot = gr.Plot(label="Correlation Heatmap")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧮 Summary Statistics")
            stats_table = gr.HighlightedText(
                label="Central Tendency & Dispersion",
                combine_adjacent=True,
                show_legend=True
            )
            raw_data_view = gr.DataFrame(label="Dataset Preview")

    # Logic: File Upload
    def on_file_upload(file):
        if file is None: return gr.update(), gr.update(choices=[]), None
        df = pd.read_csv(file.name)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        return df, gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None), engine.get_correlation_matrix(df)

    # Logic: Sample Data
    def on_load_sample():
        # Using a reliable remote URL for sample data since local data might not be ready
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        df = pd.read_csv(url)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        return df, gr.update(choices=numeric_cols, value=numeric_cols[0]), engine.get_correlation_matrix(df)

    # Logic: Analysis Update
    def update_analysis(df, column):
        if df is None or column is None or column == "":
            return None, None, []
        
        ct = engine.compute_central_tendency(df, column)
        disp = engine.compute_dispersion(df, column)
        fig_hist, fig_box = engine.generate_eda_plots(df, column)
        
        # Format stats for HighlightedText
        display_stats = [
            (f"Mean: {ct['Mean']:.2f}", "Central Tendency"),
            (f"Median: {ct['Median']:.2f}", "Central Tendency"),
            (f"Mode: {ct['Mode']:.2f}", "Central Tendency"),
            (f"Std Dev: {disp['Std Dev']:.2f}", "Dispersion"),
            (f"Variance: {disp['Variance']:.2f}", "Dispersion"),
            (f"Range: {disp['Range']:.2f}", "Dispersion"),
        ]
        
        return fig_hist, fig_box, display_stats

    # Event Handlers
    file_input.change(on_file_upload, inputs=[file_input], outputs=[raw_data_view, column_selector, corr_plot])
    load_sample_btn.click(on_load_sample, outputs=[raw_data_view, column_selector, corr_plot])
    
    run_analysis_btn.click(
        update_analysis, 
        inputs=[raw_data_view, column_selector], 
        outputs=[hist_plot, box_plot, stats_table]
    )

    column_selector.change(
        update_analysis, 
        inputs=[raw_data_view, column_selector], 
        outputs=[hist_plot, box_plot, stats_table]
    )
