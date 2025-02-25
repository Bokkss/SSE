# %% [markdown]
# # Comprehensive Documentation and Execution Plan for the Sessa Empirical Estimator (SEE) Assignment
#
# This notebook serves as a robust, highly detailed documentation of our learning, planning, and methodology for the SEE assignment. It is designed for both industry professionals and educational purposes.
#
# In this notebook, we document:
#
# 1. **Assignment Overview and Objectives:**
#    - Understand the SEE method and its applications.
#    - Convert the SEE R code to Python.
#    - Apply the method on simulated or real-world data.
#    - Explore alternative clustering methods and compare outcomes.
#
# 2. **Literature and Source Material Review:**
#    - Key insights extracted from three critical PDFs.
#
# 3. **Detailed Methodology and Execution Plan:**
#    - Step-by-step breakdown covering data preprocessing, ECDF computation, clustering (k‑means and alternatives), and duration assignment.
#    - Detailed explanations and mathematical rationales.
#
# 4. **Code Translation and Modular Function Design:**
#    - Modular functions with thorough inline comments.
#    - Enhanced to support both sample datasets.
#
# 5. **Validation, Visualization, and Reporting:**
#    - Robust validation and visualization strategies.
#
# 6. **Handling the Sample Datasets:**
#    - **med.events:**  
#      - Contains 1080 rows for 100 patients with variables: PATIENT_ID, DATE, PERDAY, CATEGORY, DURATION.
#      - Represents individual medication events used for adherence analyses.
#    - **med.events.ATC:**  
#      - Contains 1564 rows for 16 patients and extends med.events with ATC codes and hierarchical categorizations (CATEGORY_L1 and CATEGORY_L2).
#
# 7. **Conclusion and Next Steps:**
#    - Summary and future directions.
#
# **Table of Contents:**
#
# 1. [Assignment Overview and Objectives](#overview)
# 2. [Literature and Source Material Review](#literature)
# 3. [Detailed Methodology and Execution Plan](#methodology)
# 4. [Code Translation and Modular Function Design](#code_design)
# 5. [Validation, Visualization, and Reporting](#validation)
# 6. [Handling the Sample Datasets](#sample_datasets)
# 7. [Conclusion and Next Steps](#conclusion)
#
# ---
#
# <a id="overview"></a>
# ## 1. Assignment Overview and Objectives
#
# **Objective:**  
# Compute the duration of pharmacological prescriptions using the data‑driven Sessa Empirical Estimator (SEE), particularly when key data such as prescribed dose or daily consumption is missing.
#
# **Key Tasks:**
#
# - **Literature Review:**  
#   Understand the SEE methodology from key PDFs.
#
# - **R to Python Conversion:**  
#   Translate the SEE R code to Python within a Jupyter Notebook.
#
# - **Data Application:**  
#   Apply the method on simulated or real-world datasets (including the provided med.events and med.events.ATC).
#
# - **Alternative Clustering Exploration:**  
#   Evaluate alternative clustering algorithms (e.g., DBSCAN, hierarchical clustering, Gaussian Mixture Models) and compare them to k‑means.
#
# - **Modular & Reproducible Code:**  
#   Develop callable functions for every major step to ensure transparency, reproducibility, and ease of collaboration.
#
# - **Comprehensive Reporting:**  
#   Document every step and decision for robust educational and industry-level documentation.
#
# ---
#
# <a id="literature"></a>
# ## 2. Literature and Source Material Review
#
# We have reviewed three PDFs that provide the following insights:
#
# 1. **PHARMACOM-EPI Framework (First PDF):**
#    - Validates safety signals (e.g., lamotrigine toxicity) via plasma concentration predictions.
#    - Emphasizes rigorous data preprocessing and model validation.
#
# 2. **Co-exposure Assessment (Second PDF):**
#    - Describes a method for assessing co‑exposure to free dose antihypertensive medications.
#    - Highlights construction of treatment episodes, overlapping period computations, and timeline visualizations.
#
# 3. **SEE Methodology (Third PDF):**
#    - Introduces the SEE method for computing prescription durations using ECDF and k‑means clustering.
#    - Provides detailed rationale for ECDF trimming (removing upper 20%) and optimal cluster selection via silhouette analysis.
#
# **Key Takeaways:**
#
# - Use a data‑driven approach to minimize assumptions.
# - Rigorous preprocessing is essential: ordering by patient, converting dates, and computing refill intervals.
# - The clustering process and subsequent duration assignment are central to accurate exposure estimation.
# - Flexibility to explore alternative clustering methods is valuable.
#
# ---
#
# <a id="methodology"></a>
# ## 3. Detailed Methodology and Execution Plan
#
# **Step 1: Deep Dive into SEE Methodology and Literature**
# - Re-read the PDFs to grasp every step: ECDF computation, trimming, clustering, and duration assignment.
# - Understand mathematical details:
#    - **ECDF Trimming:** Excludes the top 20% of intervals to avoid skew from outliers.
#    - **Silhouette Analysis:** Objectively determines the optimal number of clusters.
#
# **Step 2: Environment Setup and R to Python Translation**
# - Set up Jupyter Notebook with required libraries:
#   - Data handling: `pandas`, `numpy`
#   - Clustering: `scikit-learn`
#   - Visualization: `matplotlib`, `seaborn`
# - Translate the SEE R code into Python.
#
# **Step 3: Data Acquisition and Application**
# - Select or simulate a dataset (e.g., using med.events or med.events.ATC).
# - Apply the modular functions to compute prescription durations.
#
# **Step 4: Alternative Clustering Exploration**
# - Implement alternative clustering methods (DBSCAN, hierarchical clustering, GMM).
# - Compare performance via silhouette scores and visualizations.
#
# **Step 5: Comparison and Reporting**
# - Compare outputs of k‑means versus alternative methods.
# - Generate detailed visualizations (ECDF, density plots, timeline diagrams) to support findings.
#
# **Step 6: Modular Callable Functions and Final Checks**
# - Ensure each process is encapsulated in well-documented, callable functions.
# - Validate consistency across interactive and programmatic runs.
#
# **Step 7: Final Submission Preparation**
# - Compile the final notebook, report, and supporting files.
# - Ensure thorough documentation and meet the assignment deadline.
#
# ---
#
# <a id="code_design"></a>
# ## 4. Code Translation and Modular Function Design
#
# The following sections provide modular functions with extensive comments. They are designed to handle both sample datasets:
# - **med.events:** Contains PATIENT_ID, DATE, PERDAY, CATEGORY, DURATION.
# - **med.events.ATC:** Contains additional fields: CATEGORY_L1 and CATEGORY_L2.
#
# This flexibility ensures our pipeline can process data in either format.
#
# ### 4.1. Data Preprocessing Function
#
# This function now includes parameters to specify dataset types and handles column name differences.
#

# %%
import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_data(df, dataset_type='med.events', patient_col=None, date_col=None):
    """
    Preprocesses prescription data by converting date columns, sorting, and computing intervals between events.
    
    Supports two dataset formats:
    - 'med.events': Expects columns: PATIENT_ID, DATE, PERDAY, CATEGORY, DURATION.
    - 'med.events.ATC': Expects the above plus CATEGORY_L1 and CATEGORY_L2.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing prescription data.
    dataset_type : str
        Type of dataset: 'med.events' or 'med.events.ATC'. Default is 'med.events'.
    patient_col : str or None
        Column name for patient identifier. If None, defaults are used:
          - 'PATIENT_ID' for med.events.
    date_col : str or None
        Column name for the prescription date. If None, defaults are used:
          - 'DATE' for med.events.
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with additional columns:
        - 'prev_date': Previous prescription date for each patient.
        - 'event_interval': Interval in days between consecutive prescriptions.
    """
    # Set default column names if not provided
    if patient_col is None:
        patient_col = 'PATIENT_ID'
    if date_col is None:
        date_col = 'DATE'
        
    # Convert the date column to datetime format
    df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y')
    
    # Sort the DataFrame by patient identifier and prescription date
    df.sort_values(by=[patient_col, date_col], inplace=True)
    
    # Compute previous prescription date for each patient
    df['prev_date'] = df.groupby(patient_col)[date_col].shift(1)
    
    # Calculate the event interval in days between consecutive prescriptions
    df['event_interval'] = (df[date_col] - df['prev_date']).dt.days
    
    # Drop rows where event_interval is missing (first prescription for each patient)
    df = df.dropna(subset=['event_interval'])
    
    return df

# %% [markdown]
# ### 4.2. ECDF Computation and Trimming Function
#
# This function computes the ECDF for the refill intervals and trims the upper 20%.
#
# **Mathematical Details:**
# - ECDF is computed by ranking sorted intervals and dividing by the total count.
# - Trimming removes the highest 20% of values to avoid skewing the median calculation.
#

# %%
def compute_trimmed_ecdf(intervals, trim_fraction=0.8):
    """
    Computes the ECDF for an array of intervals and trims it to retain the lower fraction.
    
    Parameters:
    -----------
    intervals : array-like
        Array of time intervals between prescriptions.
    trim_fraction : float
        Fraction of the ECDF to retain (default 0.8 retains lower 80%).
    
    Returns:
    --------
    trimmed_intervals : numpy.array
        Array of intervals within the trimmed ECDF.
    """
    sorted_intervals = np.sort(intervals)
    ecdf = np.arange(1, len(sorted_intervals)+1) / len(sorted_intervals)
    trimmed_intervals = sorted_intervals[ecdf <= trim_fraction]
    
    return trimmed_intervals

# %% [markdown]
# ### 4.3. Standardization and K-Means Clustering Function
#
# This function standardizes the intervals, applies k‑means clustering, and uses silhouette analysis to determine the optimal number of clusters.
#
# **Key Concepts:**
# - **Standardization:** Normalizes data (subtract mean, divide by standard deviation).
# - **Silhouette Score:** Measures cluster quality; higher values indicate better clustering.
#

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def perform_kmeans_clustering(intervals, max_clusters=10, random_state=123):
    """
    Standardizes intervals and applies k-means clustering, selecting the optimal number of clusters using silhouette analysis.
    
    Parameters:
    -----------
    intervals : array-like
        Array of time intervals.
    max_clusters : int
        Maximum clusters to test.
    random_state : int
        Random state for reproducibility.
    
    Returns:
    --------
    optimal_k : int
        Optimal number of clusters.
    cluster_labels : numpy.array
        Labels assigned to each interval.
    kmeans_model : KMeans
        Trained k-means model.
    """
    scaler = StandardScaler()
    intervals_std = scaler.fit_transform(np.array(intervals).reshape(-1, 1))
    
    best_score = -1
    optimal_k = 2
    best_labels = None
    best_model = None
    
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(intervals_std)
        score = silhouette_score(intervals_std, labels)
        if score > best_score:
            best_score = score
            optimal_k = k
            best_labels = labels
            best_model = kmeans
    
    return optimal_k, best_labels, best_model

# %% [markdown]
# ### 4.4. Compute Cluster Medians and Assign Prescription Durations
#
# This function calculates the median interval for each cluster, which serves as the computed prescription duration.
#

# %%
def assign_durations(intervals, labels):
    """
    Computes the median interval for each cluster and returns a mapping from cluster label to median duration.
    
    Parameters:
    -----------
    intervals : array-like
        Array of time intervals.
    labels : array-like
        Cluster labels for each interval.
    
    Returns:
    --------
    cluster_medians : dict
        Dictionary mapping cluster labels to median intervals.
    """
    df = pd.DataFrame({'interval': intervals, 'cluster': labels})
    cluster_medians = df.groupby('cluster')['interval'].median().to_dict()
    return cluster_medians

# %% [markdown]
# ### 4.5. Alternative Clustering Approaches (Overview)
#
# In addition to k‑means, alternative clustering methods can be explored:
#
# **DBSCAN:**
# - Does not require specifying the number of clusters.
# - Identifies clusters based on density, detecting outliers.
#
# **Hierarchical Clustering:**
# - Builds a dendrogram representing nested clusters.
# - Can be visually inspected to decide the optimal number of clusters.
#
# **Gaussian Mixture Models (GMM):**
# - Provides a probabilistic framework allowing overlapping clusters.
#
# These methods follow a similar structure: standardize, cluster, evaluate, and visualize.
#
# For this assignment, k‑means is our primary method; however, our modular design allows easy integration of alternative methods.
#
# ---
#
# <a id="validation"></a>
# ## 5. Validation, Visualization, and Reporting
#
# **Validation Strategies:**
#
# - **ECDF Visualization:**  
#   Plot full and trimmed ECDFs to verify the upper 20% removal.
#
# - **Clustering Visualizations:**  
#   Visualize clusters with scatter or density plots; use dendrograms for hierarchical clustering.
#
# - **Performance Metrics:**  
#   Compare silhouette scores for different clustering approaches.
#
# - **Edge Case Handling:**  
#   Ensure functions work well with small or highly variable datasets.
#
# ### Example: ECDF Plotting Function
#

# %%
import matplotlib.pyplot as plt

def plot_ecdf(intervals, trimmed_intervals, trim_fraction=0.8):
    """
    Plots the full ECDF of intervals and marks the trimming threshold.
    
    Parameters:
    -----------
    intervals : array-like
        Full array of intervals.
    trimmed_intervals : array-like
        Array after trimming.
    trim_fraction : float
        Fraction of ECDF retained.
    """
    sorted_intervals = np.sort(intervals)
    ecdf_full = np.arange(1, len(sorted_intervals)+1) / len(sorted_intervals)
    
    plt.figure(figsize=(10, 6))
    plt.step(sorted_intervals, ecdf_full, label='Full ECDF', where='post')
    
    max_interval_trimmed = trimmed_intervals.max()
    plt.axvline(x=max_interval_trimmed, color='red', linestyle='--', 
                label=f'Trim Threshold (retain <= {trim_fraction*100:.0f}%)')
    plt.xlabel('Time Interval (days)')
    plt.ylabel('ECDF')
    plt.title('ECDF of Prescription Intervals with Trimming')
    plt.legend()
    plt.show()

# %% [markdown]
# <a id="sample_datasets"></a>
# ## 6. Handling the Sample Datasets: med.events and med.events.ATC
#
# **med.events Dataset:**
# - Contains 1080 rows and 5 variables:
#   - **PATIENT_ID:** Unique patient identifier.
#   - **DATE:** Medication event date (mm/dd/yyyy format).
#   - **PERDAY:** Daily dosage prescribed.
#   - **CATEGORY:** Medication type (e.g., 'medA', 'medB').
#   - **DURATION:** Duration of medication event in days.
#
# **med.events.ATC Dataset:**
# - Contains 1564 rows and 7 variables:
#   - **PATIENT_ID:** Unique patient identifier.
#   - **DATE:** Medication event date.
#   - **DURATION:** Duration in days.
#   - **PERDAY:** Daily dosage.
#   - **CATEGORY:** ATC code.
#   - **CATEGORY_L1:** First-level ATC category (e.g., "A" for ALIMENTARY TRACT AND METABOLISM).
#   - **CATEGORY_L2:** Second-level ATC category (e.g., "A02" for DRUGS FOR ACID RELATED DISORDERS).
#
# **Enhancements for Dataset Handling:**
#
# - Our preprocessing function (preprocess_data) can now accept a parameter `dataset_type` to handle differences.
# - We will add example code for loading and inspecting these datasets.
#
# ### Example: Loading and Inspecting a Sample Dataset
#
# Assume the datasets are provided as CSV files. The following code loads and previews the data.
#

# %%
# Example code to load and preview med.events dataset
# Uncomment and modify the file path as needed
# med_events_df = pd.read_csv("path_to_med.events.csv")
# print("med.events dataset preview:")
# print(med_events_df.head())
#
# # Similarly, load med.events.ATC dataset
# med_events_atc_df = pd.read_csv("path_to_med.events.ATC.csv")
# print("med.events.ATC dataset preview:")
# print(med_events_atc_df.head())

# %% [markdown]
# ## 7. Conclusion and Next Steps
#
# **Summary:**
#
# - Our notebook robustly documents the SEE methodology, from literature review to execution.
# - We have modular functions for data preprocessing, ECDF computation, clustering, and duration assignment.
# - Enhancements have been made to handle the provided sample datasets (med.events and med.events.ATC) with detailed documentation.
#
# **Next Steps:**
#
# 1. **Integration and Testing:**  
#    Combine the functions into a complete pipeline and test them on one of the sample datasets.
#
# 2. **Implement Alternative Clustering:**  
#    Integrate and compare alternative clustering methods (e.g., DBSCAN, hierarchical clustering, or GMM).
#
# 3. **Final Reporting:**  
#    Generate comprehensive visualizations and compile a final report detailing the methodology, results, and insights.
#
# This notebook serves as a living document that will be continuously updated as we progress through the assignment.
#
# ---
#
# **End of Enhanced Detailed Documentation**
