"""
Main Entry Point for SyllabusML Interactive Dashboard.
Orchestrates the multi-tab interface using Gradio Blocks.
"""
import gradio as gr
from syllabusml.config import get_config
from syllabusml.stats import create_stats_tab
from syllabusml.regression import create_regression_tab
from syllabusml.classification import create_classification_tab
from syllabusml.clustering import create_clustering_tab
from syllabusml.theory import create_theory_tab

def launch_app():
    """
    Constructs and launches the multi-tab Gradio application.
    """
    # Load configuration
    cfg = get_config()
    
    # Initialize the Blocks object
    with gr.Blocks(title=cfg.app.title) as demo:
        gr.Markdown(f"# 📚 {cfg.app.title}\nInteractive Machine Learning Laboratory & Theory Explorer")

        # Tab 1: Statistics & EDA (Lab 1-3)
        with gr.Tab("📊 Statistics & EDA (Lab 1-3)"):
            create_stats_tab()

        # Tab 2: Regression (Lab 4-5)
        with gr.Tab("📈 Regression (Lab 4-5)"):
            create_regression_tab()

        # Tab 3: Classification (Lab 6-8, 10)
        with gr.Tab("🎯 Classification (Lab 6-8, 10)"):
            create_classification_tab()

        # Tab 4: Clustering (Lab 9)
        with gr.Tab("🧬 Clustering (Lab 9)"):
            create_clustering_tab()

        # Tab 5: Theory Visuals
        with gr.Tab("💡 Theory Visuals & Concepts"):
            create_theory_tab()

        gr.Markdown("---\n*Built with SyllabusML - Production-grade ML Systems Engineering Architecture.*")

    # Optimized port selection for Windows networking stability
    ports_to_try = [7860, 7861, 7862, 7863, 7864, 7865]
    
    for port in ports_to_try:
        try:
            print(f"[*] Attempting to launch on port {port}...")
            demo.launch(
                server_name="127.0.0.1",
                server_port=port,
                debug=cfg.app.debug,
                share=False,
                theme=gr.themes.Soft(),
                show_error=True,
                quiet=False
            )
            return # Success
        except (OSError, Exception) as e:
            print(f"[!] Port {port} failed: {e}")
            continue

    # Final fallback if all preferred ports fail
    print("[!] Preferred ports busy. Using system-allocated port...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=None, 
        debug=cfg.app.debug,
        share=False,
        theme=gr.themes.Soft()
    )

def main():
    """Main entry point for command-line execution."""
    launch_app()

if __name__ == "__main__":
    main()
