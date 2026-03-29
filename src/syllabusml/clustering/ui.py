"""
Gradio UI components for the Clustering Tab.
"""
import gradio as gr
import pandas as pd
from syllabusml.clustering.engine import ClusteringEngine

def create_clustering_tab():
    engine = ClusteringEngine()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Dataset Setup")
            file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
            load_data_btn = gr.Button("Load Sample (Iris)", variant="secondary")
            features = gr.CheckboxGroup(label="Select Features for Clustering")
            
            gr.Markdown("### ⚙️ K-Means Settings")
            k_slider = gr.Slider(2, 10, value=3, step=1, label="Number of Clusters (k)")
            run_btn = gr.Button("🚀 Run K-Means", variant="primary")
            elbow_btn = gr.Button("📐 Compute Elbow Chart")

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Clustering Results")
            silhouette_view = gr.Number(label="Silhouette Score (Closer to 1 is better)")
            
            with gr.Tabs():
                with gr.TabItem("2D Visualization"):
                    with gr.Row():
                        x_axis = gr.Dropdown(label="X-Axis for Plot")
                        y_axis = gr.Dropdown(label="Y-Axis for Plot")
                    cluster_plot = gr.Plot()
                with gr.TabItem("Elbow Method"):
                    elbow_plot = gr.Plot()
                with gr.TabItem("Clustered Data"):
                    data_view = gr.DataFrame()

    def handle_file_upload(file):
        if file is None: return None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
        df = pd.read_csv(file.name)
        # Include objects for financial data (cleaned in engine)
        cols = df.select_dtypes(include=['number', 'object']).columns.tolist()
        return df, gr.update(choices=cols, value=cols[:3] if len(cols) >= 3 else cols), gr.update(choices=cols, value=cols[0] if cols else None), gr.update(choices=cols, value=cols[1] if len(cols) > 1 else (cols[0] if cols else None))

    def load_data():
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        df = pd.read_csv(url)
        cols = df.select_dtypes(include=['number']).columns.tolist()
        return df, gr.update(choices=cols, value=cols), gr.update(choices=cols, value=cols[0]), gr.update(choices=cols, value=cols[1])

    df_state = gr.State()

    def run_clustering(df, feat_list, k):
        if not feat_list: return None, 0, gr.update(), None
        df_res, score = engine.run_kmeans(df, feat_list, k)
        return df_res, score, df_res

    def update_plot(df, x, y):
        if df is None or 'Cluster' not in df.columns: return None
        return engine.plot_clusters(df, x, y)

    def run_elbow(df, feat_list):
        if not feat_list: return None
        return engine.compute_elbow(df, feat_list)

    # Events
    file_input.change(handle_file_upload, inputs=[file_input], outputs=[df_state, features, x_axis, y_axis])
    load_data_btn.click(load_data, outputs=[df_state, features, x_axis, y_axis])
    
    run_btn.click(
        run_clustering, 
        inputs=[df_state, features, k_slider], 
        outputs=[df_state, silhouette_view, data_view]
    )
    
    # Update plot automatically when axes or data change
    # Note: df_state is updated to include the 'Cluster' column after run_clustering
    x_axis.change(update_plot, inputs=[df_state, x_axis, y_axis], outputs=[cluster_plot])
    y_axis.change(update_plot, inputs=[df_state, x_axis, y_axis], outputs=[cluster_plot])
    # Also trigger plot update after clustering runs
    run_btn.click(update_plot, inputs=[df_state, x_axis, y_axis], outputs=[cluster_plot])

    elbow_btn.click(run_elbow, inputs=[df_state, features], outputs=[elbow_plot])
