# 🧬 Synthetic Data Generation and Evaluation Dashboard

This Streamlit-based application provides an end-to-end workflow for generating synthetic tabular data and statistically comparing it to the original dataset using modern synthetic data generation models and statistical tools.

---

## 📌 Project Overview

This project aims to simplify and standardize the workflow of synthetic data generation for data scientists and analysts. It allows users to upload a dataset, choose how to handle missing values, apply one or more synthetic data generation models, and then validate the quality of the generated synthetic data using statistical and visual analysis.

**Key Objectives:**
- Make synthetic data generation accessible with a clean UI.
- Provide reliable statistical comparisons between original and synthetic data.
- Help users visually inspect data similarity using various distribution plots.

---

## ⚙️ Features

### ✅ Data Input
- Upload any `.csv` or `.xlsx` tabular dataset.
- Automatically detects missing values.
- User-friendly dropdown to handle missing data using:
  - `Mean`
  - `Drop rows`
  - `Interpolate`
  - `Mean/Mode` (default hybrid method)

### 🧪 Synthetic Data Generation
- Select number of synthetic rows to generate.
- Choose from the following models:
  - `GaussianCopula (SDV)`
  - `CTGAN (SDV)`
  - `TVAE (SDV)`
  - `CopulaGAN (SDV)`
  - `Bootstrap Sampling`
  - `SMOTE Generation`
  - `Classification Data Generation (binary or multiclass)`
  - `DeepEcho (SDV) Time Series Data`

### 📊 Statistical Comparison Tools
Perform a comparison between the original and synthetic data using:

1. **Descriptive Statistics**  
   View min, max, mean, std, etc., side-by-side.

2. **Correlation Matrix**  
   Compare how relationships among features are preserved.

3. **Kolmogorov–Smirnov Test (KS Test)**  
   Statistically check how similar numerical feature distributions are.

4. **Categorical Analysis**  
   Visualize value counts and compare category distributions.

5. **SDV Quality Report**  
   Built-in report from the SDV package showing quality metrics like coverage, bounds, etc.

### 📈 Visual Distribution Analysis
Compare individual or grouped feature distributions via:
- Box plots
- Violin plots
- Distribution curves
- Feature relationship plots (e.g., scatter/2D histograms)

---

## 🚀 How It Works

1. **Upload Your Dataset**  
   Load any CSV file using the sidebar uploader.

2. **Handle Missing Values**  
   Choose a strategy to treat missing data if detected.

3. **Select a Synthetic Data Model**  
   Choose a model and specify how many synthetic rows to generate.

4. **Generate Synthetic Data**  
   Synthetic data is created instantly using your selected method.

5. **Analyze and Compare**  
   Use statistical metrics and visual plots to compare the synthetic dataset against the original one.

## Example Screenshots 
## Landing Page 

<img width="1272" alt="image" src="https://github.com/user-attachments/assets/72eedab5-d493-46f5-82bd-8203f47ae3f7" />

## Missing Value Handling Page 

<img width="1276" alt="image" src="https://github.com/user-attachments/assets/562a56ed-3b70-421b-b257-0834ee56fdf9" />

## Synthetic Data Generation and Downloading CSV File 

<img width="1277" alt="image" src="https://github.com/user-attachments/assets/61d66677-b849-4279-9894-11af37f6550f" />
## Descriptive Statistics Comparison 

<img width="961" alt="image" src="https://github.com/user-attachments/assets/9cd6aa8d-d5c5-4383-8be7-bf57650d9de4" />

## Correlation Matrix 

<img width="1022" alt="image" src="https://github.com/user-attachments/assets/1d170659-335c-4526-b1f4-9ba54851e891" />

## KS Test (Kolmogorov-Smirnov Test Results)

<img width="979" alt="image" src="https://github.com/user-attachments/assets/730c8d29-632f-479f-a369-7e7fc7de7b03" />

##  Synthetic Data Vault Quality Report

<img width="974" alt="image" src="https://github.com/user-attachments/assets/ff20a4fe-f25e-44b2-85cb-e411ca32a9d5" />

## Distribution Plot 

<img width="1017" alt="image" src="https://github.com/user-attachments/assets/0e812bce-3d8d-42b8-8012-9c886d6f64ba" />

## Feature Relationship Plot

<img width="1065" alt="image" src="https://github.com/user-attachments/assets/ad5f212a-ca44-4d20-81c5-e57839db2e7d" />

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - App framework
- [SDV (Synthetic Data Vault)](https://docs.sdv.dev/) - Core synthetic data generation models
- [Plotly](https://plotly.com/python/) - Interactive visualizations
- [Pandas, NumPy, SciPy] - Data manipulation and testing

---


## Future Possible works and Notes: 
1. Here, most of the Models used are trainable and possible to perform Hyperparameter Training (Based on the  dataset model might perform differently, so hyperparameter adjustment is necessary)
2. Here, most of the packages are pulled from package documentation pages
3. The same type of methodologies can be applied to unstructured data generation, such as text, using LLMs 
4. GenAI was used to develop the project 

## How to Run 
```bash
git clone https://github.com/your-username/synthetic-data-dashboard.git
pip install -r requirements.txt
streamlit run new_app.py
```
## Demo Video 

https://github.com/user-attachments/assets/5e8edb83-87f0-4765-9860-7855d89d4aad

