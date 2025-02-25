# %% [code]
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Setup logging for debugging and audit trail
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %% [markdown]
# ## Function 1: Data Preprocessing
#
# Converts date columns, sorts by patient and date,
# and computes the interval (in days) between consecutive medication events.
#

# %% [code]
def preprocess_data(df, dataset_type='med.events', patient_col=None, date_col=None):
    """
    Preprocesses the prescription data.
    
    Supports two dataset types:
      - 'med.events': Expects columns: PATIENT_ID, DATE, PERDAY, CATEGORY, DURATION.
      - 'med.events.ATC': Expects the above plus CATEGORY_L1 and CATEGORY_L2.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input prescription data.
    dataset_type : str
        'med.events' (default) or 'med.events.ATC'.
    patient_col : str or None
        Column name for patient identifier; defaults to 'PATIENT_ID'.
    date_col : str or None
        Column name for the date; defaults to 'DATE'.
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with additional columns:
          - 'prev_date': Previous medication event date.
          - 'event_interval': Interval in days between consecutive events.
    """
    try:
        # Set default column names if not provided
        if patient_col is None:
            patient_col = 'PATIENT_ID'
        if date_col is None:
            date_col = 'DATE'
        
        # Ensure the DataFrame is not empty
        if df.empty:
            logger.warning("Input DataFrame is empty. Returning empty DataFrame.")
            return df
        
        # Convert date column to datetime; assume format 'mm/dd/yyyy'
        df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')
        if df[date_col].isnull().any():
            logger.warning("Some dates could not be parsed. Check date format.")
        
        # Sort DataFrame by patient and date
        df.sort_values(by=[patient_col, date_col], inplace=True)
        
        # Compute previous date for each patient using groupby and shift
        df['prev_date'] = df.groupby(patient_col)[date_col].shift(1)
        # Calculate event_interval in days between current and previous prescription
        df['event_interval'] = (df[date_col] - df['prev_date']).dt.days
        
        # Drop rows with missing event_interval (first record for each patient)
        df = df.dropna(subset=['event_interval'])
        df['event_interval'] = df['event_interval'].astype(int)
        
        logger.info("Data preprocessing complete. Processed {} records.".format(len(df)))
        return df
    except Exception as e:
        logger.error("Error in preprocess_data: {}".format(e))
        raise

# %% [markdown]
# ### Unit Test for preprocess_data
#
# Test the preprocessing function with a small DataFrame and an edge-case (empty DataFrame).

# %% [code]
# Create a small sample DataFrame for testing
test_data = {
    'PATIENT_ID': [1, 1, 2, 2],
    'DATE': ['01/01/2020', '01/08/2020', '02/01/2020', '02/05/2020'],
    'PERDAY': [1, 1, 2, 2],
    'CATEGORY': ['medA', 'medA', 'medB', 'medB'],
    'DURATION': [7, 7, 5, 5]
}
test_df = pd.DataFrame(test_data)
processed_test_df = preprocess_data(test_df)
assert 'event_interval' in processed_test_df.columns, "Preprocessing failed: 'event_interval' not found."

# Test empty DataFrame
empty_df = pd.DataFrame()
processed_empty_df = preprocess_data(empty_df)
assert processed_empty_df.empty, "Preprocessing failed: Empty DataFrame should return empty."

logger.info("Unit tests for preprocess_data passed.")

# %% [markdown]
# ## Function 2: ECDF Computation and Trimming
#
# Computes the Empirical Cumulative Distribution Function (ECDF) of 'event_interval' values
# and trims the upper portion (default: retain lower 80%).
#

# %% [code]
def compute_trimmed_ecdf(intervals, trim_fraction=0.8):
    """
    Computes the ECDF for an array of intervals and trims it.
    
    Parameters:
    -----------
    intervals : array-like
        Array of event intervals.
    trim_fraction : float, default=0.8
        Fraction of the ECDF to retain (e.g., 0.8 retains lower 80%).
    
    Returns:
    --------
    trimmed_intervals : numpy.array
        Array of intervals that fall within the lower trim_fraction.
    """
    try:
        # Convert intervals to numpy array if not already
        intervals = np.array(intervals)
        # Handle empty array edge-case
        if intervals.size == 0:
            logger.warning("Empty intervals array provided.")
            return intervals
        
        sorted_intervals = np.sort(intervals)
        ecdf = np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals)
        trimmed_intervals = sorted_intervals[ecdf <= trim_fraction]
        logger.info("ECDF computed and trimmed; retained {} out of {} intervals.".format(
            len(trimmed_intervals), len(sorted_intervals)))
        return trimmed_intervals
    except Exception as e:
        logger.error("Error in compute_trimmed_ecdf: {}".format(e))
        raise

# %% [markdown]
# ### Unit Test for compute_trimmed_ecdf

# %% [code]
test_intervals = np.array([1, 3, 5, 7, 9])
trimmed = compute_trimmed_ecdf(test_intervals, trim_fraction=0.8)
assert trimmed.size > 0, "ECDF trimming returned an empty array unexpectedly."
logger.info("Unit tests for compute_trimmed_ecdf passed.")

# %% [markdown]
# ## Function 3: Standardization and K-Means Clustering
#
# Standardizes the intervals, performs k-means clustering,
# and uses silhouette analysis to choose the optimal number of clusters.
#

# %% [code]
def preprocess_data(df, dataset_type='med.events', patient_col=None, date_col=None):
    """
    Preprocesses the prescription data.
    
    Supports two dataset types:
      - 'med.events': Expects columns: PATIENT_ID, DATE, PERDAY, CATEGORY, DURATION.
      - 'med.events.ATC': Expects the above plus CATEGORY_L1 and CATEGORY_L2.
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with additional columns 'prev_date' and 'event_interval'.
    """
    try:
        if patient_col is None:
            patient_col = 'PATIENT_ID'
        if date_col is None:
            date_col = 'DATE'
        
        if df.empty:
            logger.warning("Input DataFrame is empty. Returning empty DataFrame.")
            return df
        
        df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')
        if df[date_col].isnull().any():
            logger.warning("Some dates could not be parsed. Check date format.")
        
        df.sort_values(by=[patient_col, date_col], inplace=True)
        df['prev_date'] = df.groupby(patient_col)[date_col].shift(1)
        df['event_interval'] = (df[date_col] - df['prev_date']).dt.days
        
        # Use .copy() to avoid SettingWithCopyWarning
        df = df.dropna(subset=['event_interval']).copy()
        df['event_interval'] = df['event_interval'].astype(int)
        
        logger.info("Data preprocessing complete. Processed {} records.".format(len(df)))
        return df
    except Exception as e:
        logger.error("Error in preprocess_data: {}".format(e))
        raise


def perform_kmeans_clustering(intervals, max_clusters=10, random_state=123):
    """
    Standardizes intervals and performs k-means clustering with silhouette analysis.
    
    Returns:
    --------
    optimal_k : int
        Optimal number of clusters.
    best_labels : numpy.array
        Cluster labels for each interval.
    best_model : KMeans
        Fitted k-means model.
    """
    try:
        intervals = np.array(intervals)
        if intervals.size == 0:
            raise ValueError("Intervals array is empty.")
        
        scaler = StandardScaler()
        intervals_std = scaler.fit_transform(intervals.reshape(-1, 1))
        
        best_score = -1
        optimal_k = 2
        best_labels = None
        best_model = None
        
        # Ensure k is between 2 and (n_samples - 1)
        max_possible = min(max_clusters, len(intervals) - 1) if len(intervals) > 2 else 2
        
        for k in range(2, max_possible + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(intervals_std)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(intervals_std, labels)
            logger.debug("Silhouette score for k={}: {:.4f}".format(k, score))
            if score > best_score:
                best_score = score
                optimal_k = k
                best_labels = labels
                best_model = kmeans
        
        logger.info("Optimal clusters determined: {} clusters with silhouette score: {:.4f}".format(optimal_k, best_score))
        return optimal_k, best_labels, best_model
    except Exception as e:
        logger.error("Error in perform_kmeans_clustering: {}".format(e))
        raise


# %% [markdown]
# ### Unit Test for perform_kmeans_clustering

# %% [code]
# Use the previously trimmed test intervals for clustering test
optimal_k, labels, model = perform_kmeans_clustering(trimmed, max_clusters=5)
assert optimal_k >= 2, "Optimal k should be at least 2."
assert labels is not None, "Cluster labels should not be None."
logger.info("Unit tests for perform_kmeans_clustering passed.")

# %% [markdown]
# ## Function 4: Compute Cluster Medians and Assign Prescription Durations
#
# Computes the median interval for each cluster, which is used as the computed prescription duration.
#

# %% [code]
def assign_durations(intervals, labels):
    """
    Computes the median event interval for each cluster.
    
    Parameters:
    -----------
    intervals : array-like
        Array of event intervals.
    labels : array-like
        Cluster labels for each interval.
    
    Returns:
    --------
    cluster_medians : dict
        Mapping from cluster label to median interval.
    """
    try:
        df_intervals = pd.DataFrame({'interval': intervals, 'cluster': labels})
        if df_intervals.empty:
            logger.warning("No data available for computing medians.")
            return {}
        cluster_medians = df_intervals.groupby('cluster')['interval'].median().to_dict()
        logger.info("Computed medians for {} clusters.".format(len(cluster_medians)))
        return cluster_medians
    except Exception as e:
        logger.error("Error in assign_durations: {}".format(e))
        raise

# %% [markdown]
# ### Unit Test for assign_durations

# %% [code]
medians = assign_durations(trimmed, labels)
assert isinstance(medians, dict) and len(medians) > 0, "Failed to compute cluster medians."
logger.info("Unit tests for assign_durations passed.")

# %% [markdown]
# ## Integration Example: Full SEE Pipeline
#
# This function runs the full SEE pipeline:
# 1. Preprocess data.
# 2. Compute trimmed ECDF.
# 3. Perform k-means clustering.
# 4. Compute cluster medians (prescription durations).
#

# %% [code]
def run_see_pipeline(df, dataset_type='med.events'):
    """
    Runs the full SEE pipeline on the provided DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input prescription dataset.
    dataset_type : str
        'med.events' or 'med.events.ATC'.
    
    Returns:
    --------
    results : dict
        Contains:
          - preprocessed_df: DataFrame after preprocessing.
          - trimmed_intervals: Trimmed event intervals.
          - optimal_k: Optimal number of clusters.
          - cluster_labels: Cluster labels for each event.
          - cluster_medians: Mapping of clusters to median durations.
    """
    try:
        # Step 1: Preprocess the data
        preprocessed_df = preprocess_data(df, dataset_type=dataset_type)
        intervals = preprocessed_df['event_interval'].values
        
        # Check if intervals array is empty
        if intervals.size == 0:
            logger.error("No event intervals found after preprocessing.")
            return {}
        
        # Step 2: Compute trimmed ECDF
        trimmed_intervals = compute_trimmed_ecdf(intervals, trim_fraction=0.8)
        
        # Step 3: Perform k-means clustering on trimmed intervals
        optimal_k, cluster_labels, kmeans_model = perform_kmeans_clustering(trimmed_intervals)
        
        # Step 4: Compute cluster medians (prescription durations)
        cluster_medians = assign_durations(trimmed_intervals, cluster_labels)
        
        results = {
            'preprocessed_df': preprocessed_df,
            'trimmed_intervals': trimmed_intervals,
            'optimal_k': optimal_k,
            'cluster_labels': cluster_labels,
            'cluster_medians': cluster_medians
        }
        logger.info("SEE pipeline executed successfully.")
        return results
    except Exception as e:
        logger.error("Error in run_see_pipeline: {}".format(e))
        raise

# %% [markdown]
# ## Integration Test on a Simulated Dataset
#
# We now simulate a small dataset resembling med.events and run the full pipeline.
#

# %% [code]
# Create a simulated dataset for 5 patients with 5 events each
sim_data = {
    'PATIENT_ID': np.repeat(np.arange(1, 6), 5),
    'DATE': pd.date_range(start="01/01/2020", periods=25, freq='7D').strftime('%m/%d/%Y'),
    'PERDAY': np.random.randint(1, 3, 25),
    'CATEGORY': np.random.choice(['medA', 'medB'], 25),
    'DURATION': np.random.randint(5, 15, 25)
}
simulated_df = pd.DataFrame(sim_data)

# Run the pipeline
results = run_see_pipeline(simulated_df, dataset_type='med.events')

# Validate integration results
assert 'preprocessed_df' in results, "Pipeline result missing preprocessed_df."
assert results.get('trimmed_intervals', None) is not None, "Pipeline result missing trimmed_intervals."
assert results.get('optimal_k', 0) >= 2, "Optimal cluster count is less than 2."
assert isinstance(results.get('cluster_medians', None), dict), "Cluster medians not computed properly."

logger.info("Integration test on simulated dataset passed.")
logger.info("Results Summary:\nOptimal k: {}\nCluster Medians: {}".format(
    results['optimal_k'], results['cluster_medians']))

# %% [markdown]
# ## Visualization Example: Plotting ECDF
#
# This function plots the full ECDF of intervals and marks the trimmed threshold.
#

# %% [code]
def plot_ecdf(intervals, trimmed_intervals, trim_fraction=0.8):
    """
    Plots the full ECDF and indicates the trimming threshold.
    
    Parameters:
    -----------
    intervals : array-like
        Full array of event intervals.
    trimmed_intervals : array-like
        Array after trimming.
    trim_fraction : float, default=0.8
        Fraction of the ECDF retained.
    """
    sorted_intervals = np.sort(intervals)
    ecdf_full = np.arange(1, len(sorted_intervals)+1) / len(sorted_intervals)
    
    plt.figure(figsize=(10, 6))
    plt.step(sorted_intervals, ecdf_full, label='Full ECDF', where='post')
    
    # Determine the max interval retained after trimming
    max_interval_trimmed = trimmed_intervals.max() if trimmed_intervals.size > 0 else None
    if max_interval_trimmed is not None:
        plt.axvline(x=max_interval_trimmed, color='red', linestyle='--', 
                    label=f'Trim Threshold (retain <= {trim_fraction*100:.0f}%)')
    plt.xlabel('Time Interval (days)')
    plt.ylabel('ECDF')
    plt.title('ECDF of Prescription Intervals with Trimming Threshold')
    plt.legend()
    plt.show()

# %% [markdown]
# ## Next Steps
#
# 1. Integrate alternative clustering methods (e.g., DBSCAN, hierarchical clustering, GMM) and compare results.
# 2. Expand unit tests with a framework like pytest for automated testing.
# 3. Load real sample datasets (med.events, med.events.ATC) from CSV files and validate the pipeline.
# 4. Profile performance on larger datasets and optimize as needed.
#
# This enhanced pipeline provides a robust, well-documented, and maintainable conversion of the SEE R code into Python.
