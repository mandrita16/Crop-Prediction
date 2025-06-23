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

##  🚀 Clone This Repository
To clone this project to your local machine:

```bash
git clone https://github.com/your-username/your-repository-name.git

cd your-repository-name
```

Make sure to replace:
your-username with your GitHub username
your-repository-name with the actual repository name

## If you're using a virtual environment:
```bash
python -m venv crop_env
```

```bash
source crop_env/bin/activate  # On Windows: crop_env\Scripts\activate
```


## 📁 Project Structure

![image](https://github.com/user-attachments/assets/7a2c79ca-75fc-4211-a49d-75a6dbd9d72f)



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



## 🛠️ Requirements

Install required libraries in a virtual environment:
```bash
pip install -r requirements.txt
```
paste this in requirements.txt file:
```bash
numpy


pandas


scikit-learn


matplotlib


seaborn


xgboost
```

## 🚀 How to Run
Make sure you're in the crop-prediction directory and then:
```bash
python crop-prediction.py
```

📈 Sample Output
🎯 Cross-validation Accuracy (mean ± std): 96.00% ± 0.78%

Predicted crops: ['orange' 'banana' 'cotton' 'maize' ...]

📌 Future Work:
🌍 Streamlit Web App for farmer-friendly interface

📍 Google Maps API to fetch location and auto-fill soil data



⭐️ If you find this project helpful,feel free to star it on GitHub!










