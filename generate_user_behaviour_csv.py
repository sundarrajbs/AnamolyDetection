# numpy: Provides numerical operations and array handling for generating synthetic data.
# Used here to create normal and anomalous data distributions.
import numpy as np

# pandas: Offers data structures like DataFrame for data management and CSV export.
# Used here to organize generated data and save it to a CSV file.
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for dataset
n_normal = 9500  # Number of normal samples
n_anomalies = 500  # Number of anomalous samples (approx. 5% of total)
total_samples = n_normal + n_anomalies

# Generate normal user behavior data
# login_frequency: Logins per day, mean=5, std=1.5
# session_duration: Session length in minutes, mean=30, std=5
# login_hour: Hour of login (0-23), mean=14 (2 PM), std=2
normal_data = np.random.multivariate_normal(
    mean=[5, 30, 14],  # Typical user behavior
    cov=[[2.25, 0.5, 0.2], [0.5, 25, 0.1], [0.2, 0.1, 4]],  # Covariance for realistic correlations
    size=n_normal
)

# Generate anomalous user behavior data
# login_frequency: Uniformly distributed between 0 and 20 (unusual frequencies)
# session_duration: Uniformly distributed between 5 and 120 (extreme durations)
# login_hour: Uniformly distributed between 0 and 23 (odd hours)
anomaly_data = np.random.uniform(
    low=[0, 5, 0], 
    high=[20, 120, 23], 
    size=(n_anomalies, 3)
)

# Combine normal and anomalous data
data = np.vstack([normal_data, anomaly_data])

# Create DataFrame with column names
df = pd.DataFrame(
    data, 
    columns=['login_frequency', 'session_duration', 'login_hour']
)

# Save to CSV file
try:
    df.to_csv('user_behavior_large.csv', index=False)
    print(f"Generated 'user_behavior_large.csv' with {total_samples} samples ({n_anomalies} anomalies).")
except Exception as e:
    print(f"Error saving CSV: {e}")
    exit(1)