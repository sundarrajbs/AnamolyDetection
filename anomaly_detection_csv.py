# numpy: Provides numerical operations and array handling for efficient data manipulation.
# Used here for array operations on the input data.
import numpy as np

# pandas: Offers data structures like DataFrame for tabular data management and analysis.
# Used here to read the CSV file, organize data, and filter anomalies.
import pandas as pd

# scikit-learn: Machine learning library with algorithms like IsolationForest for anomaly detection.
# Used here to train and predict outliers in user behavior data.
from sklearn.ensemble import IsolationForest

# matplotlib: Visualization library for creating plots and charts.
# Used here to visualize anomalies in a scatter plot.
import matplotlib.pyplot as plt

# Load data from CSV file
# Expected CSV format: columns 'login_frequency', 'session_duration', 'login_hour'
try:
    df = pd.read_csv('user_behavior_large.csv')
    # Ensure required columns exist
    required_columns = ['login_frequency', 'session_duration', 'login_hour']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV must contain 'login_frequency', 'session_duration', and 'login_hour' columns")
except FileNotFoundError:
    print("Error: 'user_behavior.csv' not found. Please create the CSV file.")
    exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Prepare data for model
# Extract features for anomaly detection
X = df[required_columns].values

# Initialize and train Isolation Forest model
# contamination: Expected proportion of outliers (set to 5% by default)
model = IsolationForest(contamination=0.05, random_state=123)
model.fit(X)

# Predict anomalies: -1 for outliers, 1 for inliers
df['anomaly'] = model.predict(X)

# Visualize results using matplotlib
plt.figure(figsize=(10, 6))  # Set plot size
plt.scatter(df['login_frequency'], df['session_duration'], 
            c=df['anomaly'], cmap='coolwarm', alpha=0.6)  # Scatter plot with color-coded anomalies
plt.xlabel('Login Frequency (per day)')
plt.ylabel('Session Duration (minutes)')
plt.title('Anomaly Detection in User Behavior (CSV Input)')
plt.colorbar(label='Anomaly (-1) / Normal (1)')
plt.show()  # Display the plot

# Summarize results using pandas
anomalies = df[df['anomaly'] == -1]  # Filter rows marked as anomalies
print(f"Detected {len(anomalies)} anomalies out of {len(df)} samples.")
print("\nSample anomalies:")
print(anomalies.head())  # Display first few anomalies