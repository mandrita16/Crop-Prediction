# Crop-Prediction

# 🌾 Crop Recommendation System (India)

A machine learning–based crop recommendation system that suggests the best-suited crops based on soil and environmental conditions such as nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, and rainfall.

## ✅ Features

- Trained on a balanced dataset with 22 crops.
- Uses ensemble models: Random Forest, XGBoost, and Voting Classifier.
- Classification accuracy ~96%.
- Visual data exploration and preprocessing.
- Automatically encodes categorical labels and scales features.
- Cross-validation to evaluate model robustness.

## 📁 Project Structure

crop-prediction/
├── crop-prediction.py # Main script
├── crop_recommendation.csv # Dataset
├── crop_distribution.png # Label distribution plot
├── README.md 


## 🧠 Models Used

- Random Forest
- XGBoost
- Voting Classifier (ensemble)
- StandardScaler for normalization
- LabelEncoder for target processing

## 📊 Dataset

The dataset includes the following features:
- `N`, `P`, `K`: Nutrient values in soil
- `temperature`, `humidity`: Environmental factors
- `ph`: Soil pH value
- `rainfall`: Rainfall in mm
- `label`: Target crop label (e.g., rice, maize, mango, etc.)

> 📌 Note: This project currently uses a static CSV-based dataset for prediction. Streamlit UI and Google Maps integration will be added in the next version.

## 🛠️ Requirements

Install required libraries in a virtual environment:

```bash
pip install -r requirements.txt

requirements.txt:
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost

🚀 How to Run
Make sure you're in the crop-prediction directory and then:


python crop-prediction.py
📈 Sample Output
matlab

🎯 Cross-validation Accuracy (mean ± std): 96.00% ± 0.78%

Predicted crops: ['orange' 'banana' 'cotton' 'maize' ...]
