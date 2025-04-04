# main.py
# Run all steps of the KDD99 intrusion detection pipeline

import subprocess

# Step 1: Preprocessing
print("\n[STEP 1] Preprocessing data...")
subprocess.run(["python", "1_preprocessing.py"])

# Step 2: Classical Machine Learning Models
print("\n[STEP 2] Training classical ML models...")
subprocess.run(["python", "2_classical_models.py"])

# Step 3: Deep Learning Model
print("\n[STEP 3] Training deep learning model...")
subprocess.run(["python", "3_deep_learning.py"])

# Step 4: Plotting Results
print("\n[STEP 4] Plotting training performance...")
subprocess.run(["python", "4_plot_results.py"])
