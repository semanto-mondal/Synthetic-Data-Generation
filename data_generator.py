# data_generator.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer


# Import SMOTE from imbalanced-learn
from imblearn.over_sampling import SMOTE

# Import missing value handler
from missing_handler import detect_missing_values, handle_missing_values

# SDV Tabular Models
try:
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer, CopulaGANSynthesizer
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

# ydata-synthetic (Time Series)
try:
    from ydata_synthetic.synthesizers.timeseries import timegan
    TIMEGAN_AVAILABLE = True
except ImportError:
    TIMEGAN_AVAILABLE = False

# DeepEcho for time series
try:
    from deepecho import PARModel
    DEEPECHO_AVAILABLE = True
except ImportError:
    DEEPECHO_AVAILABLE = False


def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def is_categorical(series):
    return pd.api.types.is_categorical_dtype(series) or series.nunique() / len(series) < 0.05


# --- Tabular Data Generators ---

def generate_gaussian_copula_sdv(df, n_samples, missing_method="mean_mode"):
    metadata = Metadata.detect_from_dataframe(df)
    if detect_missing_values(df):
        df, _ = handle_missing_values(df, method=missing_method)

    if not SDV_AVAILABLE:
        return None, "SDV library not installed. Install with: pip install sdv"

    try:
        model = GaussianCopulaSynthesizer(metadata)
        model.fit(df)
        synthetic_df = model.sample(n_samples)
        return synthetic_df, None
    except Exception as e:
        return None, f"GaussianCopula failed: {str(e)}"


def generate_ctgan(df, n_samples, missing_method="mean_mode"):
    metadata = Metadata.detect_from_dataframe(df)
    if detect_missing_values(df):
        df, _ = handle_missing_values(df, method=missing_method)

    if not SDV_AVAILABLE:
        return None, "SDV library not installed. Install with: pip install sdv"

    try:
        model = CTGANSynthesizer(epochs=100, metadata=metadata)
        model.fit(df)
        synthetic_df = model.sample(n_samples)
        return synthetic_df, None
    except Exception as e:
        return None, f"CTGAN failed: {str(e)}"


def generate_tvae(df, n_samples, missing_method="mean_mode"):
    metadata = Metadata.detect_from_dataframe(df)
    if detect_missing_values(df):
        df, _ = handle_missing_values(df, method=missing_method)

    if not SDV_AVAILABLE:
        return None, "SDV library not installed. Install with: pip install sdv"

    try:
        model = TVAESynthesizer(epochs=100,metadata=metadata)
        model.fit(df)
        synthetic_df = model.sample(n_samples)
        return synthetic_df, None
    except Exception as e:
        return None, f"TVAE failed: {str(e)}"


def generate_copulagan(df, n_samples, missing_method="mean_mode"):
    metadata = Metadata.detect_from_dataframe(df)
    if detect_missing_values(df):
        df, _ = handle_missing_values(df, method=missing_method)

    if not SDV_AVAILABLE:
        return None, "SDV library not installed. Install with: pip install sdv"

    try:
        model = CopulaGANSynthesizer(epochs=100,metadata=metadata)
        model.fit(df)
        synthetic_df = model.sample(n_samples)
        return synthetic_df, None
    except Exception as e:
        return None, f"CopulaGAN failed: {str(e)}"


# --- Existing Methods ---

def generate_gaussian_copula(df, n_samples, missing_method="mean_mode"):
    if detect_missing_values(df):
        df, _ = handle_missing_values(df, method=missing_method)

    from numpy.linalg import cholesky

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        return None, "Need at least 2 numeric columns for Gaussian Copula"

    numeric_df = df[numeric_cols].copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    corr_matrix = np.corrcoef(scaled_data, rowvar=False)
    L = cholesky(corr_matrix)
    uncorrelated_samples = np.random.normal(size=(n_samples, len(numeric_cols)))
    correlated_samples = uncorrelated_samples @ L.T

    synthetic_df = pd.DataFrame(columns=numeric_cols)
    for i, col in enumerate(numeric_cols):
        sorted_col = np.sort(df[col])
        ecdf = np.arange(1, len(sorted_col) + 1) / len(sorted_col)
        uniform_samples = norm.cdf(correlated_samples[:, i])
        synthetic_values = np.interp(uniform_samples, ecdf, sorted_col)
        synthetic_df[col] = synthetic_values

    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in categorical_cols:
        synthetic_df[col] = np.random.choice(df[col], size=n_samples, replace=True)

    return synthetic_df, None


def generate_classification_data(df, n_samples, missing_method="mean_mode"):
    if detect_missing_values(df):
        df, _ = handle_missing_values(df, method=missing_method)

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        return None, "Need at least 2 numeric columns for classification data"

    num_features = len(numeric_cols)
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=num_features,
        n_informative=max(1, num_features - 1),
        n_redundant=0,
        n_classes=2,
        random_state=42
    )

    for i, col in enumerate(numeric_cols):
        orig_mean = df[col].mean()
        orig_std = df[col].std()
        X[:, i] = X[:, i] * orig_std + orig_mean

    synthetic_df = pd.DataFrame(X, columns=numeric_cols)

    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in categorical_cols:
        cat_probs = df[col].value_counts(normalize=True)
        synthetic_df[col] = np.random.choice(
            cat_probs.index, size=n_samples, p=cat_probs.values, replace=True
        )

    return synthetic_df, None


def generate_bootstrap(df, n_samples, missing_method="mean_mode"):
    if detect_missing_values(df):
        df, _ = handle_missing_values(df, method=missing_method)

    indices = np.random.choice(df.index, size=n_samples, replace=True)
    synthetic_df = df.iloc[indices].reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        std = df[col].std() * 0.1
        synthetic_df[col] = synthetic_df[col] + np.random.normal(0, std, size=n_samples)

    return synthetic_df, None


def generate_smote(df, n_samples, missing_method="mean_mode"):
    if detect_missing_values(df):
        df, _ = handle_missing_values(df, method=missing_method)

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        return None, "Need at least 2 numeric columns for SMOTE"

    X = df[numeric_cols]
    y = np.random.randint(0, 2, size=len(df))  # Dummy labels

    smote = SMOTE(sampling_strategy={1: n_samples}, random_state=42, k_neighbors=2)
    try:
        X_res, y_res = smote.fit_resample(X, y)
        synthetic_X = X_res[y_res == 1]
        synthetic_df = pd.DataFrame(synthetic_X, columns=numeric_cols)

        categorical_cols = df.select_dtypes(exclude=['number']).columns
        for col in categorical_cols:
            synthetic_df[col] = np.random.choice(df[col], size=len(synthetic_X), replace=True)

        return synthetic_df, None
    except Exception as e:
        return None, f"SMOTE failed: {str(e)}"


# --- Time Series Generators ---


def generate_deepecho(df, n_samples, missing_method="mean_mode"):

    def infer_data_type(series):
        if pd.api.types.is_numeric_dtype(series):
            # Check if values are all non-negative integers (for "count")
            if series.dropna().apply(lambda x: isinstance(x, (int, float)) and float(x).is_integer() and x >= 0).all():
                return 'count'
            else:
                return 'continuous'
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
            return 'categorical'
        elif pd.api.types.is_bool_dtype(series):
            return 'categorical'
        else:
            return 'unknown'
    data_types = {col: infer_data_type(df[col]) for col in df.columns}

    if detect_missing_values(df):
        df, _ = handle_missing_values(df, method=missing_method)

    if not DEEPECHO_AVAILABLE:
        return None, "deepecho library not installed. Install with: pip install deepecho"

    try:
        model = PARModel()
        model.fit(df,data_types=data_types,segment_size=1)
        synthetic_df = model.sample(n_samples)
        return synthetic_df, None
    except Exception as e:
        return None, f"DeepEcho failed: {str(e)}"


# --- Generator Registry ---

def get_available_generators():
    return {
        "Tabular Data": {
            "GaussianCopula (SDV)": lambda df, n: generate_gaussian_copula_sdv(df, n, missing_method="mean_mode"),
            "CTGAN (SDV)": lambda df, n: generate_ctgan(df, n, missing_method="mean_mode"),
            "TVAE (SDV)": lambda df, n: generate_tvae(df, n, missing_method="mean_mode"),
            "CopulaGAN (SDV)": lambda df, n: generate_copulagan(df, n, missing_method="mean_mode"),
            "Bootstrap Sampling": lambda df, n: generate_bootstrap(df, n, missing_method="mean_mode"),
            "SMOTE Generation": lambda df, n: generate_smote(df, n, missing_method="mean_mode"),
            "Classification Data": lambda df, n: generate_classification_data(df, n, missing_method="mean_mode"),
        },
        "Time Series Data": {
            #"TimeGAN (ydata-synthetic)": lambda df, n: generate_timegan(df, n, missing_method="mean_mode"),
            "DeepEcho (SDV)": lambda df, n: generate_deepecho(df, n, missing_method="mean_mode"),
        }
    }