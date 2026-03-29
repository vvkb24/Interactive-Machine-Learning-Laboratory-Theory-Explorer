"""
Gradio UI components for the Theory Visuals Tab.
"""
import gradio as gr
from syllabusml.theory.engine import TheoryEngine

def create_theory_tab():
    engine = TheoryEngine()

    with gr.Row():
        with gr.Column(scale=1):
            topic_selector = gr.Accordion("🔍 Select Theory Topic", open=True)
            with topic_selector:
                theory_topic = gr.Radio(
                    ["Perceptron & Linear Separability", "Dimensionality Reduction (PCA)", "MLP & Backpropagation", "Ensemble Methods (Intuition)", "Support Vector Machines (SVM)", "Genetic Algorithms", "Reinforcement Learning"],
                    label="Choose a concept to explore:",
                    value="Perceptron & Linear Separability"
                )
            
            # Context-sensitive Parameters
            with gr.Column(visible=True) as perceptron_params:
                w1_slider = gr.Slider(-5, 5, value=1, label="Weight 1 (w1)")
                w2_slider = gr.Slider(-5, 5, value=1, label="Weight 2 (w2)")
                bias_slider = gr.Slider(-5, 5, value=-2, label="Bias (b)")
                run_p_btn = gr.Button("🚀 Run Perceptron Simulation", variant="primary")
            
            with gr.Column(visible=False) as pca_params:
                pca_angle = gr.Slider(0, 90, value=45, label="Data Correlation Angle (Deg)")
                n_pts = gr.Slider(50, 500, value=200, step=50, label="Number of Data Points")
                run_pca_btn = gr.Button("🔍 Project Data (PCA)", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 📚 Theory Explorer & Interactive Visuals")
            
            with gr.Group():
                theory_explanation = gr.Markdown(
                    label="Academic Explanation",
                    line_breaks=True,
                    container=True,
                    height=400 # Support scrollable area
                )
            
            with gr.Group():
                gr.Markdown("#### 🎨 Interactive Visual (Selection Dependent)")
                theory_plot = gr.Plot(visible=True)
            
            # Backprop specific content (Deprecated: now in explanation)
            backprop_markdown = gr.Markdown(visible=False)

    # UI Visibility Logic
    def handle_topic_change(topic):
        p_vis = gr.update(visible=topic == "Perceptron & Linear Separability")
        pca_vis = gr.update(visible=topic == "Dimensionality Reduction (PCA)")
        plot_vis = gr.update(visible=topic in ["Perceptron & Linear Separability", "Dimensionality Reduction (PCA)"])
        bp_vis = gr.update(visible=False)
        
        explanation = engine.get_theory_explanation(topic)
        
        # Determine initial visual
        plot = None
        if topic == "Perceptron & Linear Separability":
            plot = engine.simulate_perceptron(1, 1, -2)
        elif topic == "Dimensionality Reduction (PCA)":
            plot = engine.simulate_pca_projection(200, 45)
            
        return p_vis, pca_vis, plot_vis, bp_vis, plot, explanation

    # Live Updates for Perceptron
    def update_perceptron(w1, w2, b):
        return engine.simulate_perceptron(w1, w2, b)

    # Live Updates for PCA
    def update_pca(n, angle):
        return engine.simulate_pca_projection(n, angle)

    # Event Handlers
    theory_topic.change(
        handle_topic_change, 
        inputs=[theory_topic], 
        outputs=[perceptron_params, pca_params, theory_plot, backprop_markdown, theory_plot, theory_explanation]
    )
    
    # Live sliders for Perceptron
    run_p_btn.click(update_perceptron, [w1_slider, w2_slider, bias_slider], theory_plot)
    w1_slider.change(update_perceptron, [w1_slider, w2_slider, bias_slider], theory_plot)
    w2_slider.change(update_perceptron, [w1_slider, w2_slider, bias_slider], theory_plot)
    bias_slider.change(update_perceptron, [w1_slider, w2_slider, bias_slider], theory_plot)
    
    # Live sliders for PCA
    run_pca_btn.click(update_pca, [n_pts, pca_angle], theory_plot)
    pca_angle.change(update_pca, [n_pts, pca_angle], theory_plot)
    n_pts.change(update_pca, [n_pts, pca_angle], theory_plot)
