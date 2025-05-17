# missing_handler.py

import pandas as pd
import numpy as np

def detect_missing_values(df):
    """Detect if there are any missing values in the DataFrame"""
    return df.isnull().sum().sum() > 0

def handle_missing_values(df, method="mean_mode"):
    """
    Handle missing values in a DataFrame
    
    Args:
        df: Original pandas DataFrame
        method: Method to use for imputation
            - "drop": Drop rows with missing values
            - "mean_mode": Fill numerical with mean, categorical with mode
            - "interpolate": Use linear interpolation for numerical columns
            
    Returns:
        Cleaned DataFrame
    """
    if not detect_missing_values(df):
        return df, False  # No missing values found
    
    if method == "drop":
        df_clean = df.dropna()
    elif method == "mean_mode":
        df_clean = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df_clean[col] = df[col].fillna(df[col].mean())
            else:
                df_clean[col] = df[col].fillna(df[col].mode()[0])
    elif method == "interpolate":
        df_clean = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns
        df_clean[numeric_cols] = df[numeric_cols].interpolate(method='linear', axis=0)
        
        # Fill remaining non-numeric with mode
        for col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        raise ValueError("Invalid missing value handling method")
    
    return df_clean, True  # Missing values were handled