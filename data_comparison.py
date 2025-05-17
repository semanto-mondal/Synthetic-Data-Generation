import pandas as pd
import numpy as np
from scipy import stats
from sdv.evaluation.single_table import evaluate_quality

def get_descriptive_stats(df):
    """
    Calculate descriptive statistics for numeric columns in a DataFrame
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with descriptive statistics
    """
    stats_df = df.describe().T
    
    # Add additional statistics
    if not df.empty and len(df.select_dtypes(include=['number']).columns) > 0:
        stats_df['variance'] = df.var()
        stats_df['skewness'] = df.skew()
        stats_df['kurtosis'] = df.kurtosis()
    
    return stats_df

def calculate_correlation(df):
    """
    Calculate correlation matrix for numeric columns
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Correlation matrix as DataFrame
    """
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] > 1:
        return numeric_df.corr()
    return pd.DataFrame()

def ks_test(original, synthetic, column=None):
    """
    Perform Kolmogorov-Smirnov test between two datasets
    
    Args:
        original: Original data series
        synthetic: Synthetic data series
        column: Column name (optional)
        
    Returns:
        Tuple of (statistic, p-value)
    """
    if len(original) > 0 and len(synthetic) > 0:
        stat, p = stats.ks_2samp(original, synthetic)
        return stat, p
    return None, None

def compare_datasets_ks(original_df, synthetic_df):
    """
    Compare original and synthetic datasets using KS test for all numeric columns
    
    Args:
        original_df: Original pandas DataFrame
        synthetic_df: Synthetic pandas DataFrame
        
    Returns:
        DataFrame with KS test results
    """
    numeric_cols = original_df.select_dtypes(include=['number']).columns
    ks_results = []
    
    for col in numeric_cols:
        stat, p = ks_test(original_df[col], synthetic_df[col], col)
        ks_results.append({
            'Feature': col,
            'KS Statistic': stat,
            'p-value': p,
            'Similarity': 'High' if p > 0.05 else 'Medium' if p > 0.01 else 'Low'
        })
    
    return pd.DataFrame(ks_results)

def compare_categorical_distributions(original_df, synthetic_df, column):
    """
    Compare categorical distributions between original and synthetic data
    
    Args:
        original_df: Original pandas DataFrame
        synthetic_df: Synthetic pandas DataFrame
        column: Column name to compare
        
    Returns:
        DataFrame with combined value counts
    """
    # Get value counts for both datasets
    orig_counts = original_df[column].value_counts(normalize=True).reset_index()
    orig_counts.columns = ['value', 'proportion']
    orig_counts['dataset'] = 'Original'
    
    synth_counts = synthetic_df[column].value_counts(normalize=True).reset_index()
    synth_counts.columns = ['value', 'proportion']
    synth_counts['dataset'] = 'Synthetic'
    
    # Calculate absolute difference
    merged_counts = pd.merge(
        orig_counts[['value', 'proportion']], 
        synth_counts[['value', 'proportion']], 
        on='value', 
        suffixes=('_orig', '_synth'),
        how='outer'
    ).fillna(0)
    
    merged_counts['abs_diff'] = abs(merged_counts['proportion_orig'] - merged_counts['proportion_synth'])
    
    # Add to combined dataframe
    combined_counts = pd.concat([orig_counts, synth_counts])
    
    return combined_counts, merged_counts['abs_diff'].mean()

def compare_all_categorical(original_df, synthetic_df):
    """
    Compare all categorical columns between datasets
    
    Args:
        original_df: Original pandas DataFrame
        synthetic_df: Synthetic pandas DataFrame
        
    Returns:
        DataFrame with category similarity scores
    """
    # Find categorical columns
    cat_cols = [col for col in original_df.columns 
              if col in synthetic_df.columns and
              (pd.api.types.is_categorical_dtype(original_df[col]) or 
               original_df[col].nunique() / len(original_df) < 0.05)]
    
    results = []
    for col in cat_cols:
        _, mean_diff = compare_categorical_distributions(original_df, synthetic_df, col)
        similarity = 1 - mean_diff  # Convert difference to similarity (0-1)
        similarity_category = 'High' if similarity > 0.9 else 'Medium' if similarity > 0.7 else 'Low'
        
        results.append({
            'Feature': col,
            'Similarity Score': similarity,
            'Similarity': similarity_category
        })
    
    return pd.DataFrame(results)

def compare_correlation_difference(original_df, synthetic_df):
    """
    Compare correlation matrices between original and synthetic data
    
    Args:
        original_df: Original pandas DataFrame
        synthetic_df: Synthetic pandas DataFrame
        
    Returns:
        DataFrame with correlation differences
    """
    corr_orig = calculate_correlation(original_df)
    corr_synth = calculate_correlation(synthetic_df)
    
    if not corr_orig.empty and not corr_synth.empty:
        # Ensure both matrices have the same columns
        common_cols = list(set(corr_orig.columns) & set(corr_synth.columns))
        corr_diff = corr_orig[common_cols].loc[common_cols] - corr_synth[common_cols].loc[common_cols]
        return corr_diff
    
    return "No Difference in Correlation Matrices"

def generate_sdv_quality_report(real_data, synthetic_data,metadata):
    """
    Generates an SDV quality report including:
    - Overall score
    - Column shape scores
    - Column pair trend scores
    """
    try:
      

        # Evaluate quality
        quality_report = evaluate_quality(real_data, synthetic_data, metadata)

        # Get components
        overall_score = quality_report.get_score()
        column_shapes = quality_report.get_details('Column Shapes')
        column_pair_trends = quality_report.get_details('Column Pair Trends')

        return {
            'overall_score': overall_score,
            'column_shapes': column_shapes,
            'column_pair_trends': column_pair_trends
        }

    except Exception as e:
        return {"error": str(e)}