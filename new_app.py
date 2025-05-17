import streamlit as st
import pandas as pd
import numpy as np
import io
from sdv.metadata import Metadata
# Import from our custom modules
from data_generator import get_available_generators, is_categorical, is_numeric
from data_comparison import (
    get_descriptive_stats, 
    calculate_correlation, 
    compare_datasets_ks,
    compare_categorical_distributions,
    compare_all_categorical,
    compare_correlation_difference,
    generate_sdv_quality_report
)
from data_visualization import (
    plot_feature_histogram,
    plot_feature_kde,
    plot_categorical_comparison,
    plot_scatter_comparison,
    plot_density_comparison,
    plot_correlation_matrix,
    plot_ks_test_results,
    plot_parallel_coordinates
)

# Set page configuration
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stPlotlyChart {
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        border-radius: 5px;
        padding: 10px;
    }
    .stPlotlyChart:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>Synthetic Data Generator</h1>", unsafe_allow_html=True)
st.markdown("""
This application allows you to generate synthetic data from your uploaded dataset. 
You can select from multiple generation methods, compare statistical metrics, and visualize distributions.
""")

# Initialize session state
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'synthetic_df' not in st.session_state:
    st.session_state.synthetic_df = None
if 'generation_warning' not in st.session_state:
    st.session_state.generation_warning = None

# Sidebar for controls
st.sidebar.header("Controls")

# File upload section
st.sidebar.markdown("<div class='section-header'>1. Upload Data</div>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Process uploaded file
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Detect missing values
        from missing_handler import detect_missing_values, handle_missing_values

        has_missing = detect_missing_values(df)

        if has_missing:
            st.markdown("<h3 style='color:#FF5722;'>Missing Values Detected</h3>", unsafe_allow_html=True)
            st.warning("The dataset contains missing values. Please select a method to handle them.")

            missing_method = st.selectbox(
                "Select a method to handle missing values:",
                ["mean_mode", "interpolate", "drop"]
            )

            if st.button("Clean Data"):
                with st.spinner("Handling missing values..."):
                    cleaned_df, was_handled = handle_missing_values(df, method=missing_method)
                    if was_handled:
                        st.session_state.original_df = cleaned_df
                        st.success("Missing values handled successfully!")
                        st.dataframe(cleaned_df.head(), use_container_width=True)
                    else:
                        st.error("Failed to handle missing values.")
        else:
            st.success("No missing values found in the dataset.")
            st.session_state.original_df = df
            st.dataframe(df.head(), use_container_width=True)

    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")


# Data generation options
st.sidebar.markdown("<div class='section-header'>2. Generate Synthetic Data</div>", unsafe_allow_html=True)

if st.session_state.original_df is not None:
    # Get all available generators (now grouped)
    generators_by_category = get_available_generators()

    # Flatten the generator dict into a single list
    generators_flat = {}
    for category, methods in generators_by_category.items():
        generators_flat.update(methods)

    # Let user choose from the flattened list
    generation_method = st.sidebar.selectbox(
        "Select Generation Method",
        list(generators_flat.keys())
    )
    
    n_samples = st.sidebar.slider(
        "Number of synthetic samples", 
        min_value=10, 
        max_value=max(1000, len(st.session_state.original_df) * 2),
        value=len(st.session_state.original_df),
        step=10
    )
    
    if st.sidebar.button("Generate Synthetic Data"):
        with st.spinner("Generating synthetic data..."):
            # Get the selected generator function
            generator_func = generators_flat[generation_method]
            
            # Generate synthetic data using the selected method
            st.session_state.synthetic_df, st.session_state.generation_warning = generator_func(
                st.session_state.original_df, 
                n_samples
            )
            
            if st.session_state.synthetic_df is not None:
                st.sidebar.success(f"Generated {len(st.session_state.synthetic_df)} synthetic samples")
                if st.session_state.generation_warning:
                    st.sidebar.warning(st.session_state.generation_warning)
            else:
                if st.session_state.generation_warning:
                    st.sidebar.error(st.session_state.generation_warning)
                else:
                    st.sidebar.error("Failed to generate synthetic data")

# Main display area
if st.session_state.original_df is not None:
    # Display dataset tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📉 Statistical Comparison", "🔄 Distribution Visualization"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='section-header'>Original Data</div>", unsafe_allow_html=True)
            st.dataframe(st.session_state.original_df.head(10), use_container_width=True)
            
            if st.button("Download Original Data"):
                csv = st.session_state.original_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="original_data.csv",
                    mime="text/csv",
                )
        
        with col2:
            st.markdown("<div class='section-header'>Synthetic Data</div>", unsafe_allow_html=True)
            if st.session_state.synthetic_df is not None:
                st.dataframe(st.session_state.synthetic_df.head(10), use_container_width=True)
                
                if st.button("Download Synthetic Data"):
                    csv = st.session_state.synthetic_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="synthetic_data.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Generate synthetic data using the controls in the sidebar")
    
    with tab2:
        if st.session_state.synthetic_df is not None:
            st.markdown("<div class='section-header'>Statistical Metrics Comparison</div>", unsafe_allow_html=True)
            
            metric_type = st.selectbox(
                "Select Metric Type",
                ["Descriptive Statistics", "Correlation Matrix", "KS Test", "Categorical Analysis", "SDV Report"]
            )
            
            if metric_type == "Descriptive Statistics":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Data Statistics**")
                    original_stats = get_descriptive_stats(st.session_state.original_df.select_dtypes(include=['number']))
                    st.dataframe(original_stats, use_container_width=True)
                
                with col2:
                    st.markdown("**Synthetic Data Statistics**")
                    synthetic_stats = get_descriptive_stats(st.session_state.synthetic_df.select_dtypes(include=['number']))
                    st.dataframe(synthetic_stats, use_container_width=True)
            
            elif metric_type == "Correlation Matrix":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Data Correlation**")
                    corr_orig = calculate_correlation(st.session_state.original_df)
                    if not corr_orig.empty:
                        fig = plot_correlation_matrix(corr_orig, "Original Data Correlation Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least 2 numeric columns for correlation matrix")
                
                with col2:
                    st.markdown("**Synthetic Data Correlation**")
                    corr_synth = calculate_correlation(st.session_state.synthetic_df)
                    if not corr_synth.empty:
                        fig = plot_correlation_matrix(corr_synth, "Synthetic Data Correlation Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least 2 numeric columns for correlation matrix")
                
                # Show correlation difference if both matrices exist
                corr_diff = compare_correlation_difference(st.session_state.original_df, st.session_state.synthetic_df)
                if not corr_diff.empty:
                    st.markdown("**Correlation Difference (Original - Synthetic)**")
                    fig = plot_correlation_matrix(corr_diff, "Correlation Difference Matrix")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif metric_type == "KS Test":
                st.markdown("**Kolmogorov-Smirnov Test Results**")
                st.markdown("""
                The Kolmogorov-Smirnov test compares the distributions of original and synthetic data.
                - Lower statistic values indicate more similar distributions
                - Higher p-values (>0.05) suggest distributions are not significantly different
                """)
                
                # Run KS tests on all numeric columns
                ks_results = compare_datasets_ks(st.session_state.original_df, st.session_state.synthetic_df)
                
                if not ks_results.empty:
                    st.dataframe(ks_results, use_container_width=True)
                    
                    # Plot KS test results
                    fig = plot_ks_test_results(ks_results)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numeric features found for KS test")
            
            elif metric_type == "Categorical Analysis":
                # Get categorical columns
                cat_cols = [col for col in st.session_state.original_df.columns 
                          if col in st.session_state.synthetic_df.columns and
                          is_categorical(st.session_state.original_df[col])]
                
                if cat_cols:
                    # Show overall similarity scores
                    cat_similarity = compare_all_categorical(st.session_state.original_df, st.session_state.synthetic_df)
                    
                    st.markdown("**Categorical Features Similarity Scores**")
                    st.dataframe(cat_similarity, use_container_width=True)
                    
                    # Allow user to select a categorical feature to analyze in detail
                    selected_cat = st.selectbox("Select categorical feature for detailed analysis", cat_cols)
                    
                    # Compare distributions of the selected categorical feature
                    combined_counts, _ = compare_categorical_distributions(
                        st.session_state.original_df, 
                        st.session_state.synthetic_df, 
                        selected_cat
                    )
                    
                    # Plot categorical comparison
                    fig = plot_categorical_comparison(combined_counts, selected_cat)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No categorical features found in the dataset")

            elif metric_type == "SDV Report":
                st.markdown("<h3>📊 Synthetic Data Vault Quality Report</h3>", unsafe_allow_html=True)
                st.markdown("""
                The SDV report evaluates synthetic data using two main metrics:
                - **Column Shapes**: How well each feature's distribution is preserved
                - **Column Pair Trends**: How well relationships between features are maintained
                """)

                with st.spinner("Generating SDV quality report..."):
                    report_data = generate_sdv_quality_report(
                        st.session_state.original_df,
                        st.session_state.synthetic_df,
                        metadata=Metadata.detect_from_dataframe(st.session_state.original_df)
                    )

                    if isinstance(report_data, dict) and "error" in report_data:
                        st.error(f"❌ Error generating SDV report: {report_data['error']}")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📈 Column Shapes Score", f"{report_data['column_shapes']['Score'].mean() * 100:.2f}%")
                        with col2:
                            st.metric("📉 Column Pair Trends Score", f"{report_data['column_pair_trends']['Score'].mean() * 100:.2f}%")

                        st.metric("🏆 Overall Quality Score", f"{report_data['overall_score'] * 100:.2f}%")

                        # --- Column Shapes Details ---
                        st.markdown("### 🔍 Column Shape Matching")
                        st.dataframe(report_data['column_shapes'], use_container_width=True)

                        # --- Column Pair Trends Details ---
                        #st.markdown("### 🔄 Column Pair Trend Matching")
                        #st.dataframe(report_data['column_pair_trends'], use_container_width=True)
                        
        else:
            st.info("Generate synthetic data to see statistical comparisons")
    
    with tab3:
        if st.session_state.synthetic_df is not None:
            st.markdown("<div class='section-header'>Distribution Visualization</div>", unsafe_allow_html=True)
            
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Distributions", "Feature Relationships"]
            )
            
            if viz_type == "Distributions":
                # Select columns to visualize
                numeric_cols = st.session_state.original_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Select feature to visualize", numeric_cols)
                    
                    # Plot histogram comparison
                    hist_fig = plot_feature_histogram(
                        st.session_state.original_df, 
                        st.session_state.synthetic_df, 
                        selected_col

                    )
                    st.plotly_chart(hist_fig, use_container_width=True)
                    
                    # Plot KDE comparison
                    kde_fig = plot_feature_kde(
                        st.session_state.original_df, 
                        st.session_state.synthetic_df, 
                        selected_col
                    )
                    st.plotly_chart(kde_fig, use_container_width=True)
                
                # Categorical columns visualization
                cat_cols = [col for col in st.session_state.original_df.columns 
                          if col in st.session_state.synthetic_df.columns and
                          is_categorical(st.session_state.original_df[col])]
                
                if len(cat_cols) > 0:
                    selected_cat_col = st.selectbox("Select categorical feature to visualize", cat_cols)
                    
                    # Compare categorical distributions
                    combined_counts, _ = compare_categorical_distributions(
                        st.session_state.original_df, 
                        st.session_state.synthetic_df, 
                        selected_cat_col
                    )
                    
                    # Plot categorical comparison
                    cat_fig = plot_categorical_comparison(combined_counts, selected_cat_col)
                    st.plotly_chart(cat_fig, use_container_width=True)
            
            elif viz_type == "Feature Relationships":
                numeric_cols = st.session_state.original_df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_col = st.selectbox("Select X-axis feature", numeric_cols)
                    
                    with col2:
                        y_col = st.selectbox("Select Y-axis feature", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    # Plot scatter comparison
                    scatter_fig = plot_scatter_comparison(
                        st.session_state.original_df, 
                        st.session_state.synthetic_df, 
                        x_col, 
                        y_col
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)
                    
                    # Plot density comparison
                    density_fig = plot_density_comparison(
                        st.session_state.original_df, 
                        st.session_state.synthetic_df, 
                        x_col, 
                        y_col
                    )
                    st.plotly_chart(density_fig, use_container_width=True)
                
                else:
                    st.info("Need at least 2 numeric columns for feature relationship visualization")
        
        else:
            st.info("Generate synthetic data to see visualizations")

else:
    # App landing page when no data is uploaded
    st.info("Please upload a dataset using the sidebar to get started.")
    st.markdown("### 🧪 Available Synthetic Data Generation Models")

    st.markdown("""
    #### 📊 Tabular Data Models

    - **GaussianCopula (SDV)**: Uses Gaussian copulas to model feature-wise distributions and their dependencies.
    - **CTGAN (SDV)**: GAN-based model tailored for tabular data with imbalanced and mixed-type features.
    - **TVAE (SDV)**: Variational Autoencoder model for capturing non-linear dependencies in tabular data.
    - **CopulaGAN (SDV)**: Combines Copula modeling with GAN to preserve both marginal and joint distributions.
    - **Bootstrap Sampling**: Generates new samples by randomly resampling (with replacement) from the original data.
    - **SMOTE Generation**: Creates synthetic samples by interpolating between nearest neighbors (used for class balancing).
    - **Classification Data**: Custom generator for binary or multi-class datasets, preserving class-wise structure.

    #### ⏱️ Time Series Models

    - **DeepEcho (SDV)**: RNN-based model for time series generation that preserves temporal dependencies and context.
    """)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Step 1: Upload Data
        - Upload a CSV or Excel file with your original data
        - The app will analyze the features and prepare for synthesis
        """)
    
    with col2:
        st.markdown("""
        ### Step 2: Generate Synthetic Data
        - Choose from multiple generation methods
        - Adjust the number of samples to generate
        - Each method has different strengths and applications
        """)
    
    with col3:
        st.markdown("""
        ### Step 3: Analyze & Visualize
        - Compare statistical metrics between original and synthetic data
        - Visualize distributions to assess quality
        - Download your synthetic data for use in projects
        """)

# Add footer with information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Synthetic Data Generator | Created with Streamlit</p>
    <p>This tool provides various methods to generate synthetic data that preserves statistical properties of the original dataset.</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # This will execute when the script is run directly
    pass
