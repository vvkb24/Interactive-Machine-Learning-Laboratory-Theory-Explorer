# 👋 SyllabusML – Complete Interactive ML Lab & Theory Explorer

# SyllabusML

I created this repo because staring at boring lecture notes and 47 broken Jupyter notebooks was slowly killing my soul.

Like any sane student, I ditched the boring PDFs and built this multi-tab Gradio app.

It has everything: Mean, Median, Mode, House Price predictions, Decision Trees that love to overfit, and even some theory visualizations that sometimes make sense.

**Current status**: It works on my machine. Sometimes.

This is 100% open source.  
If you also suffer from the same syllabus, feel free to use it.  
If you're a pro ML Engineer and you're laughing at my code — congratulations, you're correct. Send a PR and fix my life.

Built with existential crisis and coffee.

### **Key Features**
- **Unified Dashboard**: A sleek, multi-tab interface using `gr.Blocks`.
- **Production Layout**: Adheres to the `src/` layout pattern for clean packaging.
- **Hermetic Dependencies**: Powered by `uv` for lightning-fast and reproducible environments.
- **Config-Driven**: Uses **Hydra** to decouple environment settings from core logic.
- **Interactive Visuals**: Kinetic Plotly charts for high-dimensional data exploration.

---

## 🛠️ Tech Stack

- **Core Engine**: `Python 3.11+`, `NumPy`, `Pandas`, `SciPy`
- **Machine Learning**: `Scikit-Learn`, `Statsmodels`, `Joblib`
- **Frontend/UI**: `Gradio` (v4.0+)
- **Visualization**: `Plotly`, `Matplotlib`, `Seaborn`
- **Configuration**: `Hydra`, `OmegaConf`
- **Environment/Build**: `uv`, `pyproject.toml`

---

## 📂 Project Structure

```text
SyllabusML/
├── src/
│   └── syllabusml/           # Main Package
│       ├── stats/            # Lab 1-3: Descriptive Statistics & EDA
│       ├── regression/       # Lab 4-5: Simple & Multiple Linear Regression
│       ├── classification/   # Lab 6-8, 10: Supervised Learning & Mini-Project
│       ├── clustering/       # Lab 9: Unsupervised Learning (K-Means)
│       ├── theory/           # Interactive Theory Visuals (Neural Nets, PCA)
│       ├── utils/            # Loaders, Persistance, and UI Helpers
│       ├── config.py         # Hydra Configuration Loader
│       └── app.py            # Gradio Orchestrator
├── configs/                  # YAML-based Environment & Model Configs
├── data/                     # Raw and Processed Datasets
├── artifacts/                # Serialized Models (.joblib)
├── notebooks/                # Exploratory Research
├── tests/                    # Unit and Integration Tests
├── pyproject.toml            # Project Metadata & Dependencies
└── run.py                    # Application Entry Point
```

### **Design Decisions**
- **`src/` Layout**: Prevents accidental imports of the local folder and ensures the package is tested as an installed entity.
- **Modular Sub-packages**: Each lab is isolated. The logic (`engine.py`) is strictly separated from the presentation (`ui.py`).

---

## ⚙️ Installation & Setup

### **1. Prerequisites**
Install [uv](https://github.com/astral-sh/uv) (recommended) or use standard `pip`.

### **2. Environment Setup**
```bash
# Clone the repository
git clone <your-repo-url>
cd SyllabusML

# Install dependencies and sync environment
uv sync
```

### **3. Running the Application**
```bash
# Launch the Gradio Dashboard
uv run python run.py
```
The app will be available at `http://127.0.0.1:7860`.

---

## 📖 Syllabus Coverage Mapping

| Lab/Theory Point | Description | Module / Tab |
|:---|:---|:---|
| **Lab 1-2** | Central Tendency, Dispersion, NumPy/SciPy | `stats/` (Tab 1) |
| **Lab 3** | Pandas & Matplotlib for ML Applications | `stats/` (Tab 1) |
| **Lab 4** | Simple Linear Regression | `regression/` (Tab 2) |
| **Lab 5** | Multiple Regression (House Price Prediction) | `regression/` (Tab 2) |
| **Lab 6** | Decision Tree + Parameter Tuning | `classification/` (Tab 3) |
| **Lab 7** | K-Nearest Neighbors (KNN) | `classification/` (Tab 3) |
| **Lab 8** | Logistic Regression | `classification/` (Tab 3) |
| **Lab 9** | K-Means Clustering (Elbow & Silhouette) | `clustering/` (Tab 4) |
| **Lab 10** | **Mini Project**: Classifier Performance Report | `classification/` (Tab 3) |
| **Theory** | Perceptron, Backpropagation, PCA, Ensembles | `theory/` (Tab 5) |

---

## 🏗️ How SyllabusML Works

### **Architecture Overview**

SyllabusML follows a **layered modular architecture**:

```
┌─────────────────────────────────────────────────────┐
│        Gradio Dashboard (app.py)                    │
│    Multi-tab User Interface & Event Handlers        │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│        Module UI Layers (stats/ui.py, etc.)         │
│    Layout, Sliders, Buttons, Event Bindings         │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│     Module Engine Layers (stats/engine.py, etc.)    │
│  ML Logic, Data Processing, Model Training, Metrics │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│  Utilities & Configuration (utils, config.py)       │
│ Data Loading, Model Persistence, Hydra Config       │
└─────────────────────────────────────────────────────┘
```

### **Data Flow**
1. **User uploads/selects data** → Gradio receives as file or selects sample
2. **Data is parsed** → `utils.load_dataset()` converts to Pandas DataFrame
3. **Engine processes** → Statistical, ML, or visualization logic runs
4. **Results displayed** → Plotly charts, metrics, confusion matrices rendered
5. **Models saved** → Trained artifacts stored in `artifacts/` as `.joblib` files

---

## 📚 Detailed Tab-by-Tab Usage Guide

### **Tab 1: 📊 Statistics & EDA (Lab 1-3)**

**What it covers:**
- Mean, Median, Mode (Central Tendency)
- Variance, Standard Deviation, Range (Dispersion)
- Data Distribution Visualization
- Correlation Analysis

**How to use:**
1. Click **"Load Sample (Iris)"** to use the built-in Iris dataset, or upload your own CSV.
2. Select a numeric column from the **"Select Column for Analysis"** dropdown.
3. View the **Histogram** and **Box Plot** tabs to analyze distribution.
4. Check the **"Summary Statistics"** table for central tendency and dispersion metrics.
5. Examine the **"Correlation Heatmap"** to understand feature relationships.

**Expected Output:**
- Histograms showing data distribution with frequency bars
- Box plots revealing outliers (points beyond whiskers)
- Summary statistics formatted as: Mean: 5.84, Median: 5.90, Std Dev: 0.83, etc.
- Color-coded correlation matrix (blue = positive, red = negative)

**Dataset Format:**
- CSV file with numeric and categorical columns
- Missing values should be represented as `NaN` (will be dropped automatically)
- At least 10 rows recommended for meaningful statistics

---

### **Tab 2: 📈 Regression (Lab 4-5)**

**What it covers:**
- Simple Linear Regression (1 feature)
- Multiple Linear Regression (multiple features)
- House Price Prediction (Boston/California Housing dataset)

**How to use:**
1. Click **"Load Boston/California Housing"** or upload your own CSV.
2. **Select Target Variable (Y)**: Usually a continuous column like price or salary.
3. **Select Features (X)**: Choose numeric columns that predict the target.
4. Adjust **"Test Size"** (default: 0.2 = 80/20 split) and **"Random State"** for reproducibility.
5. Click **🚀 Train Regression Model** to fit the model.
6. Review **Performance Metrics**:
   - **R² Score**: How well the model explains variance (closer to 1 is better)
   - **RMSE**: Root Mean Squared Error in target units
   - **MAE**: Mean Absolute Error
7. Examine the **"Actual vs Predicted"** scatter plot: Points near the diagonal line = good predictions.
8. Check **"Model Coefficients"** to see feature weights.

**Expected Output:**
- R² between 0 and 1 (e.g., 0.72 means 72% of variance explained)
- RMSE and MAE in the same units as your target (e.g., $)
- Scatter plot showing perfect predictions along the diagonal line
- Coefficients table showing which features have the strongest influence

**Best Practices:**
- Scale features if they have very different ranges
- Ensure test_size is ≥ 0.2 to have enough test data (at least 20+ samples)
- If R² is too low (<0.5), try adding more relevant features or investigate data quality

---

### **Tab 3: 🎯 Classification (Lab 6-8, 10)**

**What it covers:**
- Decision Tree Classifier (Lab 6)
- K-Nearest Neighbors / KNN (Lab 7)
- Logistic Regression (Lab 8)
- **Mini Project (Lab 10)**: Automated comparison of 4 classifiers

**How to use (Individual Lab):**
1. Click **"Load Iris (Classification)"** or upload a CSV with a target column.
2. **Select Target Variable**: A categorical column (e.g., species, label, class).
3. **Select Features**: Numeric columns used for prediction.
4. **Choose Algorithm** from the radio buttons.
5. **Adjust Hyperparameters**:
   - **Decision Tree**: Max Depth (higher = more complex tree)
   - **KNN**: Number of Neighbors (typically 3-7)
6. Click **🚀 Train Classifier**.
7. Review **Confusion Matrix**: Shows True Positives, False Positives, etc.
8. Check **Performance Metrics**:
   - **Accuracy**: % of correct predictions
   - **Precision**: % of positive predictions that were correct
   - **Recall**: % of actual positives that were found
   - **F1 Score**: Harmonic mean of Precision and Recall

**How to use (Mini Project - Lab 10):**
1. Load data and select features/target as above.
2. Switch to the **"Mini Project (Lab 10)"** sub-tab.
3. Click **📊 Run Comparison Report**.
4. The system trains 4 models (Decision Tree, KNN, Logistic Regression, Random Forest) automatically.
5. Compare **Performance Table** and **Accuracy Bar Chart**.
6. **Conclusion**: The algorithm with the highest F1 Score is often best-performing.

**Expected Confusionmatrix Output:**
```
        Predicted-A  Predicted-B
Actual-A     45           5        (45 TP, 5 FN)
Actual-B      3          47        (3 FP, 47 TP)
```

**Best Practices:**
- Use at least 100+ samples for meaningful classification
- Ensure classes are somewhat balanced (if 95% class-A, 5% class-B, use accuracy with caution)
- Try multiple algorithms; rarely is there a "one-size-fits-all" classifier

---

### **Tab 4: 🧬 Clustering (Lab 9)**

**What it covers:**
- K-Means Clustering (unsupervised learning)
- Elbow Method (finding optimal k)
- Silhouette Analysis (cluster quality)

**How to use:**
1. Click **"Load Sample (Iris)"** or upload your CSV.
2. **Select Features for Clustering**: Choose numeric columns.
3. Click **📐 Compute Elbow Chart**:
   - Look for the "elbow" point where WCSS stops decreasing sharply
   - For Iris, optimal k is usually 3
4. Set the **"Number of Clusters (k)"** slider to the elbow point.
5. Click **🚀 Run K-Means**.
6. **Check Silhouette Score** (ranges -1 to 1):
   - Close to 1: Clusters are well-separated
   - Close to 0: Overlapping clusters
   - Close to -1: Points assigned to wrong cluster
7. Select **X-Axis** and **Y-Axis** in the "2D Visualization" tab to view clusters.
8. Each color represents a different cluster.

**Expected Output:**
- Elbow chart showing decreasing WCSS; elbow typically at k=3 for Iris
- Silhouette score ~0.55 indicates reasonable clustering
- 2D scatter plot with 3 distinct colored regions

**Best Practices:**
- Don't go beyond k = number_of_features; k=5-10 is usually sufficient
- Normalize features before clustering (done automatically in engine)
- Use silhouette score as a quality metric; aim for > 0.5

---

### **Tab 5: 💡 Theory Visuals & Concepts**

**What it covers:**
- **Perceptron & Linear Separability**: Interactive visualization of decision boundaries
- **Dimensionality Reduction (PCA)**: See how high-dimensional data projects to lower dimensions
- **Backpropagation Intuition**: Mathematical formulas for neural network learning
- **Ensemble Methods**: Conceptual overview

**How to use (Perceptron):**
1. Select **"Perceptron & Linear Separability"** from the dropdown.
2. Adjust sliders:
   - **Weight 1 (w1)**: Coefficient for feature 1
   - **Weight 2 (w2)**: Coefficient for feature 2
   - **Bias (b)**: Decision boundary offset
3. Watch the contour plot update in real-time.
4. Red and blue regions represent the two classes.
5. The **black dashed line** is the decision boundary.

**How to use (PCA):**
1. Select **"Dimensionality Reduction (PCA)"**.
2. Adjust:
   - **Data Correlation Angle**: How aligned features are
   - **Number of Data Points**: Sample size
3. Blue dots = original data; Red dots = PCA-projected data.
4. The variance explained is shown in the title (e.g., 0.92 = 92% of variance retained in 1D).

**Educational Value:**
- Understand how changing weights rotates the decision boundary
- See how PCA identifies the direction of maximum variance
- Read the Backpropagation formulas to understand how neural networks learn

---

## ⚙️ Configuration Guide

The file `configs/config.yaml` controls global settings:

```yaml
data:
  raw_dir: "data/raw"              # Where raw datasets are stored
  processed_dir: "data/processed"  # Where cleaned data goes
  
model:
  random_state: 42                 # For reproducibility
  test_size: 0.2                   # Train-test split (80-20 default)
  artifacts_dir: "artifacts"       # Where models are saved
  
training:
  cv_folds: 5                      # Cross-validation folds
  verbose: true                    # Print debug info
  
app:
  port: 7860                       # Gradio server port
  debug: true                      # Enable debug mode
```

**How to customize:**
- Change `random_state` to get different train-test splits (for sensitivity analysis)
- Increase `test_size` to 0.3 if you have plenty of data
- Change `port` if 7860 is already in use

---

## 📤 How to Upload Custom Datasets

### **Dataset Requirements**
- **Format**: CSV file (comma or semicolon separated)
- **Columns**: At least one numeric column + one target column
- **Size**: 10-10,000 rows recommended
- **Missing Values**: Represented as blank, `NaN`, or `N/A` (will be auto-removed)

### **Example Dataset Structure**
```csv
feature1,feature2,feature3,target
5.1,3.5,1.4,0
7.0,3.2,4.7,1
6.3,3.3,6.0,2
```

### **Dataset Upload Steps**
1. Prepare your CSV file locally
2. Open the SyllabusML app at `http://127.0.0.1:7860`
3. Go to the relevant tab (Stats, Regression, Classification, or Clustering)
4. Click the file upload button
5. Select your CSV file
6. The app auto-detects columns; select features and target accordingly

---

## 🔧 Troubleshooting & Common Issues

### **Issue: Port 7860 already in use**
```bash
# Change port in configs/config.yaml:
# app:
#   port: 7861
```

### **Issue: "ModuleNotFoundError: No module named 'gradio'"**
```bash
# Reinstall dependencies
uv sync
```

### **Issue: "Non-numeric column selected"**
- The engine automatically filters non-numeric columns
- If still failing, ensure your CSV has at least 2 numeric columns

### **Issue: Low accuracy in Classification**
- Try more features
- Increase training data size
- Tune hyperparameters (max_depth for trees, n_neighbors for KNN)

### **Issue: Regression R² too low**
- Add more relevant features
- Check for outliers in your data
- Verify target and features have a linear relationship

### **Issue: Clustering Silhouette Score negative**
- Reduce number of clusters (k)
- Rerun with a different random state
- Check if data naturally clusters (not all datasets cluster well)

---

## 🚀 Best Practices & Tips

1. **Always start with EDA (Tab 1)**
   - Understand your data distribution before modeling
   - Look for outliers and missing values

2. **Use consistent random_state**
   - Ensures reproducibility across runs
   - Great for research papers and presentations

3. **Test/Validation First**
   - Train on training set, evaluate on test set
   - Never train and test on the same data

4. **Try multiple models (Classification)**
   - Use the Mini Project to compare 4 classifiers
   - Different problems favor different algorithms

5. **Interpret metrics carefully**
   - Accuracy alone is misleading for imbalanced data
   - Use Precision, Recall, and F1 Score together

6. **Save your best models**
   - Models are automatically saved to `artifacts/`
   - Use them later for predictions

---

## 💡 Example Workflows

### **Workflow 1: Predicting House Prices**
1. **Tab 1**: Load Boston Housing data, analyze price distribution
2. **Tab 2**: Load same data, select price as target, select features (rooms, age, etc.)
3. **Train**: Check R² ≈ 0.7-0.8
4. **Interpret**: View coefficient weights to see which features matter most

### **Workflow 2: Flower Classification**
1. **Tab 1**: Load Iris, analyze feature distributions
2. **Tab 3**: Load Iris, try individual algorithms (Decision Tree first)
3. **Tab 3**: Switch to Mini Project to compare all 4 classifiers
4. **Conclusion**: Random Forest typically wins on Iris (but sometimes KNN is close)

### **Workflow 3: Understanding Data Clusters**
1. **Tab 1**: Load any dataset, check feature correlations
2. **Tab 4**: Run Elbow Method to find optimal clusters
3. **Tab 4**: Set k to elbow point, visualize clusters in 2D
4. **Theory (Tab 5)**: Use PCA visualization to see how many dimensions needed

---

## 🧠 Engineering Highlights

### **1. Configuration Management**
We treat hyperparameters and paths as data, not code. By using **Hydra**, you can override any setting (e.g., `test_size`, `random_state`) via `configs/config.yaml` without touching the source.

### **2. Mathematical Correctness**
- **Vectorization**: Large-scale statistical computations avoid Python loops, utilizing NumPy's C-level optimizations.
- **Normalization**: Clustering logic includes automatic Z-score standardization to ensure Euclidean distances are meaningful.

### **3. Failure Mode Handling**
- **Input Validation**: The UI gracefully handles non-numeric columns and empty dataframes.
- **State Management**: Uses `gr.State` to ensure data persistence across different UI interactions safely.

---

## � How to Extend SyllabusML

### **Adding a New Lab Module**
1. Create a new folder under `src/syllabusml/` (e.g., `src/syllabusml/anomaly_detection/`)
2. Create two files:
   - `engine.py`: Your ML logic (model training, metrics)
   - `ui.py`: Gradio UI components
3. Export in `__init__.py`:
   ```python
   from syllabusml.anomaly_detection.ui import create_anomaly_tab
   ```
4. Import and add to `app.py`:
   ```python
   with gr.Tab("🎯 Anomaly Detection"):
       create_anomaly_tab()
   ```

### **Adding Custom Utilities**
Add helper functions to `src/syllabusml/utils/__init__.py`:
```python
def custom_preprocessing(df):
    # Your logic here
    return processed_df
```

### **Modifying Configuration**
Edit `configs/config.yaml` to add new settings:
```yaml
my_module:
  threshold: 0.5
  min_samples: 10
```
Load in code:
```python
from syllabusml.config import get_config
cfg = get_config()
print(cfg.my_module.threshold)
```

---

## 🧪 Testing & Validation

### **Run Individual Module Tests**
```bash
# Test statistics module
python -m pytest tests/test_stats.py -v

# Test classification module
python -m pytest tests/test_classification.py -v
```

### **Check Code Quality**
```bash
# Format code
uv run black src/

# Lint
uv run flake8 src/
```

---

## 📊 Performance Considerations

| Operation | Time Complexity | Scalability |
|:---|:---|:---|
| **Statistics** | O(n) | Works with 1M+ rows |
| **Linear Regression** | O(n·p²) | Up to 1000+ features |
| **Decision Tree** | O(n·log(n)·p) | Up to 10K rows/features |
| **K-Means** | O(n·k·d·i) | 100K+ rows feasible |
| **Correlation Matrix** | O(p²) | Up to 500 features |

*n = samples, p = features, k = clusters, d = dimensions, i = iterations*

---

## 🤝 Contributing & Support

### **How to Contribute**
1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Make changes following the modular structure
4. Test your changes: `uv run python run.py`
5. Submit a Pull Request

### **Reporting Issues**
- Include Python version: `python --version`
- Include installed packages: `pip list`
- Provide a minimal reproducible example
- Describe expected vs. actual behavior

---

## 📝 License & Citation

If you use **SyllabusML** in your research or project, please cite:

```bibtex
@software{syllabusml2026,
  title={SyllabusML: Complete Interactive ML Lab & Theory Explorer},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/SyllabusML}
}
```

---

## 🎓 Learning Resources

### **Recommended Reading**
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Plotly Documentation](https://plotly.com/python/)
- [Gradio Quickstart](https://www.gradio.app/guides/quickstart)

### **Relevant Textbooks**
- "Introduction to Statistical Learning" by James et al.
- "Hands-On Machine Learning" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Bishop

### **Online Courses**
- Andrew Ng's Machine Learning Specialization (Coursera)
- Fast.ai Deep Learning Course
- Stanford CS229 Machine Learning Notes

---

## 🌟 Highlights & Differentiators

✅ **Why use SyllabusML?**
- ⚡ **Production-ready code**: Not typical class notebooks
- 🎯 **100% Syllabus coverage**: Every lab + theory included
- 🔧 **Easily extensible**: Add modules without touching core code
- 📊 **Beautiful visualizations**: Interactive Plotly charts
- 🔄 **Reproducible**: Config-driven, fixed random seeds
- 📦 **Clean architecture**: `src/` layout, separation of concerns
- 🚀 **Fast to launch**: No dependency conflicts with `uv`

---

## 👤 Author & Purpose

**SyllabusML** was developed as a complete, portfolio-ready ML Engineering project covering university ML syllabus. It demonstrates production-grade Python packaging, modular design, and practical ML engineering standards suitable for:
- College/University course submissions
- Job interviews & technical portfolios
- Research & exploratory analysis
- Teaching ML concepts interactively

---

## 📞 Contact & Questions

For questions or feedback, please:
- Open an Issue on GitHub
- Check existing documentation in this README
- Review the code comments in `src/syllabusml/*/engine.py` for detailed logic explanations

---

*Built with ❤️ for the ML Community.*
*Last updated: March 29, 2026*

