"""
Shared utility functions for data loading, model persistence, and UI helpers.
"""
import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Optional

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or Parquet dataset.
    
    Args:
        file_path: Path to the dataset file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file is missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    if path.suffix == ".csv":
        return pd.read_csv(file_path)
    elif path.suffix in [".parquet", ".pqt"]:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def save_model(model: Any, model_name: str, artifacts_dir: str = "artifacts") -> Path:
    """
    Serializes an ML model using joblib.
    
    Args:
        model: Any scikit-learn compatible model.
        model_name: Name for the saved file.
        artifacts_dir: Root directory for artifacts.
        
    Returns:
        Path: Path to the saved model.
    """
    save_path = Path(artifacts_dir) / f"{model_name}.joblib"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return save_path

def load_model(model_name: str, artifacts_dir: str = "artifacts") -> Any:
    """
    Load a serialized model.
    """
    load_path = Path(artifacts_dir) / f"{model_name}.joblib"
    if not load_path.exists():
        raise FileNotFoundError(f"No model found at {load_path}")
    return joblib.load(load_path)
