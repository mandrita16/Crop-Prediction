# Step 1: Import Libraries and Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random

print("✅ Libraries imported successfully!")

print("\n" + "="*50)
print("STEP 1: LOADING DATA")
print("="*50)

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")
print("✅ Dataset loaded successfully!")
print("Dataset shape:", data.shape)
print("Columns:", data.columns.tolist())

# Simulate 'month' (1 to 12) and 'region'
months = np.random.randint(1, 13, size=len(data))
data['month'] = months
data['quarter'] = data['month'].apply(lambda x: (x - 1) // 3 + 1)

# Simulate region/state for now (replace with real values via Google Maps API later)
regions = ['Punjab', 'West Bengal', 'Tamil Nadu', 'Maharashtra', 'Bihar', 'Karnataka', 'UP', 'Assam']
data['region'] = [random.choice(regions) for _ in range(len(data))]

print("✅ Synthetic 'month', 'quarter', and 'region' columns added!")

# Step 2:
print("\n" + "="*50)
print("STEP 2: DATA PREPROCESSING & EXPLORATION")
print("="*50)

# First 5 rows
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# Plot label distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='label', data=data, order=data['label'].value_counts().index)
plt.title("Crop Distribution")
plt.tight_layout()
plt.savefig("crop_distribution.png")
print("\n📊 'crop_distribution.png' saved successfully!")

# Encoding categorical 'region' feature
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['region_encoded'] = le.fit_transform(data['region'])

# Save for prediction decoding
crop_encoder = LabelEncoder()
data['label_encoded'] = crop_encoder.fit_transform(data['label'])

# Final feature set
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'month', 'quarter', 'region_encoded']
X = data[features]
y = data['label_encoded']

# Step 3:
print("\n" + "="*50)
print("STEP 3: TRAINING & CROSS-VALIDATION")
print("="*50)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Train-test split (Stratified to balance all classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add light noise to simulate real-world variability
noise = np.random.normal(0, 0.09, X_train_scaled.shape)  # increased noise
X_train_scaled_noisy = X_train_scaled + noise

# Build a slightly less complex Random Forest
model = RandomForestClassifier(
    n_estimators=60,         # ↓ less trees
    max_depth=5,             # ↓ more restricted depth
    max_features=0.6,        # ↓ fewer features used per split
    min_samples_leaf=3,      # ↑ force broader generalization
    random_state=42
)

model.fit(X_train_scaled_noisy, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled_noisy, y_train, cv=5)
print(f"🎯 Cross-validation Accuracy (mean ± std): {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
crop_names = crop_encoder.inverse_transform(y_pred)
print("\nPredicted crops:", crop_names[:10])

print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred, target_names=crop_encoder.classes_))
