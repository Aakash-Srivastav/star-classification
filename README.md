# 🌌 Star Classification Using Machine Learning
This project applies multiple supervised learning models to classify celestial objects — GALAXY, STAR, and QSO — based on data from the Sloan Digital Sky Survey (SDSS). It includes data preprocessing, visualization, and model evaluation using techniques like Grid Search and cross-validation.

## 📂 Dataset
The dataset used is assumed to be star_classification.csv, which should include photometric features (u, g, r, i, z, redshift, alpha, delta) and object class labels.

## 🧰 Features
Exploratory Data Analysis (EDA) with Seaborn & Matplotlib

Label encoding and feature cleaning

Heatmap and distribution visualization

Redshift-based scatter plots

Train-test split with stratification

Model pipelines with StandardScaler

GridSearchCV hyperparameter tuning

Confusion matrix and classification reports

## 🧠 Models Used
Logistic Regression

Linear Discriminant Analysis (LDA)

Quadratic Discriminant Analysis (QDA)

Decision Tree Classifier

Random Forest Classifier

Support Vector Classifier (SVC)

XGBoost Classifier

## 🚀 How to Run
### 1. Install Dependencies

```
bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 2. Prepare Data
Make sure the file star_classification.csv is in the same directory.

### 3. Run the Script
```
bash
Copy
Edit
python main.py
```

Each model will train, perform grid search (where applicable), and output a classification report and confusion matrix.

## 📊 Visualizations
The script includes:

Histograms for each feature

Heatmap of feature correlation

Pair plots of main features

Scatter plots showing redshift versus magnitude per class

## 🗂️ Project Structure
```
bash
Copy
Edit
main.py                    # Main analysis and training script
star_classification.csv    # Input dataset (not included here)
README.md                  # Project description
```

## 📌 Notes
The dataset is cleaned by dropping certain ID columns and rows with invalid values (e.g., non-positive u values).

Grid search is applied to optimize model parameters where appropriate.

Results are displayed via plots and printed reports.
