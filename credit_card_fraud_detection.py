# ------------------------------
# Fraud Detection Project - Complete
# ------------------------------

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42

LABELS = ["Normal", "Fraud"]

# ------------------------------
# Step 2: Load Data
# ------------------------------
data = pd.read_csv(r'C:\Users\sneha\Downloads\creditcard.csv.zip')

print(data.head())
print(data.info())

# ------------------------------
# Step 3: Explore Data
# ------------------------------
# Class distribution
print(data['Class'].value_counts())

sns.countplot(x='Class', data=data)
plt.title("Normal vs Fraud Transactions")
plt.show()

# Separate fraud & normal transactions
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

print("Fraud shape:", fraud.shape)
print("Normal shape:", normal.shape)

# Transaction amount statistics
print(fraud['Amount'].describe())
print(normal['Amount'].describe())

# Histogram of transaction amounts
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.hist(fraud['Amount'], bins=50, color='red')
ax1.set_title("Fraud Transactions")
ax2.hist(normal['Amount'], bins=50, color='blue')
ax2.set_title("Normal Transactions")
plt.xlabel("Amount")
plt.ylabel("Number of Transactions")
plt.show()

# Scatter plot: Time vs Amount
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14,8))
ax1.scatter(fraud['Time'], fraud['Amount'], color='red', alpha=0.5)
ax1.set_title("Fraud Transactions")
ax1.set_ylabel("Amount")
ax2.scatter(normal['Time'], normal['Amount'], color='blue', alpha=0.5)
ax2.set_title("Normal Transactions")
ax2.set_xlabel("Time")
ax2.set_ylabel("Amount")
plt.show()

# Correlation Heatmap
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.title("Feature Correlation")
plt.show()

# ------------------------------
# Step 4: Prepare Data
# ------------------------------
# Sample 10% for faster processing
data_sample = data.sample(frac=0.1, random_state=RANDOM_SEED)

# Separate fraud & valid
Fraud = data_sample[data_sample['Class'] == 1]
Valid = data_sample[data_sample['Class'] == 0]

# Outlier fraction
outlier_fraction = len(Fraud) / float(len(Valid))
print("Outlier Fraction:", outlier_fraction)

# Features & target
columns = [c for c in data_sample.columns if c != 'Class']
X = data_sample[columns]
Y = data_sample['Class']

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Step 5: Build Models
# ------------------------------

# 1️⃣ Isolation Forest
model_if = IsolationForest(n_estimators=100, contamination=outlier_fraction, random_state=RANDOM_SEED)
model_if.fit(X_scaled)
y_pred_if = model_if.predict(X_scaled)
y_pred_if[y_pred_if == 1] = 0
y_pred_if[y_pred_if == -1] = 1

# 2️⃣ Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
y_pred_lof = lof.fit_predict(X_scaled)
y_pred_lof[y_pred_lof == 1] = 0
y_pred_lof[y_pred_lof == -1] = 1

# 3️⃣ One-Class SVM
ocsvm = OneClassSVM(nu=outlier_fraction, kernel='rbf', gamma=0.05)
ocsvm.fit(X_scaled)
y_pred_svm = ocsvm.predict(X_scaled)
y_pred_svm[y_pred_svm == 1] = 0
y_pred_svm[y_pred_svm == -1] = 1

# ------------------------------
# Step 6: Evaluate Models
# ------------------------------
def evaluate_model(y_true, y_pred, model_name):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\n")

evaluate_model(Y, y_pred_if, "Isolation Forest")
evaluate_model(Y, y_pred_lof, "Local Outlier Factor")
evaluate_model(Y, y_pred_svm, "One-Class SVM")

# ------------------------------
# Step 7: Compare Models
# ------------------------------
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to get key metrics
def model_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_pred)
    }

# Store metrics for all models
results = {
    "Isolation Forest": model_metrics(Y, y_pred_if),
    "Local Outlier Factor": model_metrics(Y, y_pred_lof),
    "One-Class SVM": model_metrics(Y, y_pred_svm)
}

# Convert to DataFrame for easy visualization
results_df = pd.DataFrame(results).T
results_df = results_df[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
print("\nModel Comparison:\n")
print(results_df)

# ------------------------------
# Step 8: Visualize Model Comparison
# ------------------------------
results_df.plot(kind='bar', figsize=(12,6))
plt.title("Fraud Detection Model Comparison")
plt.ylabel("Score")
plt.ylim(0,1)  # Scores range from 0 to 1
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()

results_df.to_csv("fraud_model_comparison.csv", index=True)

plt.savefig("model_comparison.png", dpi=300)  # saves the chart as an image

# Create the bar chart again
fig = results_df.plot(kind='bar', figsize=(12,6)).get_figure()
plt.title("Fraud Detection Model Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()  # ensures labels are not cut off

fig.savefig("model_comparison.png", dpi=300)  # save the image first
plt.show()  # then display it

# ------------------------------
# Save Model Comparison Bar Chart
# ------------------------------

import matplotlib.pyplot as plt

# Make sure results_df exists
# results_df should have your models as rows and metrics as columns
# Example:
# results_df = pd.DataFrame({
#     "Accuracy": [0.998, 0.995, 0.996],
#     "Precision": [0.84, 0.70, 0.65],
#     "Recall": [0.86, 0.77, 0.72],
#     "F1-Score": [0.85, 0.73, 0.68],
#     "ROC-AUC": [0.92, 0.88, 0.85]
# }, index=["Isolation Forest", "LOF", "One-Class SVM"])

# Create figure
fig = results_df.plot(kind='bar', figsize=(12,6)).get_figure()

# Add title and labels
plt.title("Fraud Detection Model Comparison")
plt.ylabel("Score")
plt.ylim(0,1)  # all scores are between 0 and 1
plt.xticks(rotation=0)
plt.legend(loc='lower right')

# Save figure BEFORE showing it
fig.savefig("model_comparison.png", dpi=300)  # saved in current working directory

# Show the plot
plt.show()

print("✅ model_comparison.png saved successfully! Check your folder.")
