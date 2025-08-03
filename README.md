# Student Score Prediction Project

## 📊 Overview

This project implements a machine learning pipeline to predict student exam scores based on various performance factors. The model analyzes 20 different features including study habits, socio-economic factors, and learning environment characteristics to predict academic performance.

## 🎯 Objective

The goal is to build accurate regression models that can predict student exam scores, helping educators and institutions understand the key factors that influence academic success and potentially identify students who may need additional support.

## 📁 Project Structure

```
Student-Score-Prediction-elevvo/
├── main.ipynb                          # Main notebook with complete analysis pipeline
├── functions.py                        # Custom utility functions for ML pipeline
├── README.md                          # Project documentation
├── data/
│   └── StudentPerformanceFactors.csv  # Dataset (downloaded via Kaggle API)
├── old_trial/
│   ├── first_trial.ipynb             # Initial experimental notebook
│   └── first_trial_functions.py      # Initial utility functions
└── __pycache__/                      # Python cache files
```

## 📈 Dataset Information

The dataset contains **6,608 student records** with the following features:

### Numerical Features (6)

- `Hours_Studied`: Weekly study hours
- `Attendance`: Attendance percentage
- `Sleep_Hours`: Average daily sleep hours
- `Previous_Scores`: Previous academic scores
- `Tutoring_Sessions`: Number of tutoring sessions attended
- `Physical_Activity`: Hours of physical activity per week

### Categorical Features (14)

#### Ordinal Features (8)

- `Parental_Involvement`: Low, Medium, High
- `Access_to_Resources`: Low, Medium, High
- `Motivation_Level`: Low, Medium, High
- `Family_Income`: Low, Medium, High
- `Teacher_Quality`: Low, Medium, High
- `Distance_from_Home`: Near, Moderate, Far
- `Peer_Influence`: Positive, Neutral, Negative
- `Parental_Education_Level`: High School, College, Postgraduate

#### Nominal Features (6)

- `Extracurricular_Activities`: Yes, No
- `Internet_Access`: Yes, No
- `Learning_Disabilities`: Yes, No
- `Gender`: Male, Female
- `School_Type`: Public, Private

### Target Variable

- `Exam_Score`: Final exam score (continuous variable)

## Quick Start

1. Clone the repository
2. Install dependencies
3. Run the main notebook:

```bash
jupyter notebook main.ipynb
```

### Running the Complete Pipeline

The `main.ipynb` notebook contains the following sections:

1. **Data Download**: Automatic dataset download from Kaggle
2. **Exploratory Data Analysis (EDA)**: Comprehensive data exploration
3. **Data Preprocessing**: Cleaning, encoding, and feature engineering
4. **Model Training**: Multiple regression algorithms
5. **Model Evaluation**: Performance metrics and comparison

## 🔍 Methodology

### 1. Exploratory Data Analysis

- **Correlation Analysis**: Heatmap visualization of feature relationships
- **Distribution Analysis**: Histograms for numerical and categorical features
- **Target Variable Analysis**: Distribution and transformation exploration
- **Missing Value Assessment**: Identification of data quality issues

### 2. Data Preprocessing Pipeline

- **Missing Value Handling**: Imputation with most frequent values
- **Categorical Encoding**:
  - Label Encoding for ordinal features
  - One-Hot Encoding for nominal features
- **Outlier Detection**: Z-score method (threshold = 3)
- **Feature Scaling**: StandardScaler for numerical features
- **Data Splitting**: Train-test split with no data leakage

### 3. Model Implementation

#### Linear Regression

- Basic linear regression model
- Serves as baseline for comparison

#### Polynomial Regression

- Degree 2 polynomial features
- Grid search for optimal hyperparameters
- Cross-validation for robust evaluation

### 4. Model Evaluation Metrics

- **R² Score**: Coefficient of determination
- **Mean Squared Error (MSE)**: Average squared prediction errors
- **Mean Absolute Error (MAE)**: Average absolute prediction errors
- **Cross-Validation**: 5-fold CV for reliable performance estimation

## 📊 Key Features

### Custom Functions (`functions.py`)

- `explore_target_transformations()`: Analyzes target variable distributions
- `train_linear_regression()`: Complete linear regression pipeline
- `train_polynomial_regression()`: Polynomial regression with grid search
- Data preprocessing utilities with train-test split awareness

### Data Leakage Prevention

- Proper train-test splitting before preprocessing
- Scaler fitting only on training data
- Missing value imputation within splits

### Comprehensive EDA

- Numerical feature distributions
- Categorical feature analysis
- Correlation matrices
- Target variable exploration

## 🎯 Results

The project implements multiple regression models with the following capabilities:

- **Baseline Linear Regression**: Simple interpretable model
- **Polynomial Regression**: Captures non-linear relationships
- **Grid Search Optimization**: Automated hyperparameter tuning
- **Cross-Validation**: Robust performance evaluation

## 🔮 Future Enhancements

- Try different models like Random Forest, XGBoost, etc. → tried in the old trial
- Feature engineering to create new features / drop features that don't contribute much like gender, school type, etc. → tried in the old trial
- Try transforming the target variable (e.g., log transformation) to see if it improves model performance → tried in the old trial
- Try regularization techniques like Lasso or Ridge to improve model stability

## 📝 Notes

- The `old_trial/` folder contains initial experimental work
- All preprocessing steps are designed to prevent data leakage
- The project uses Kaggle's API for automated dataset downloading
- Models are evaluated using multiple metrics for comprehensive assessment

## 📄 License

This project is available for educational and research purposes.

---

_This project is part of the Elevvo ML Internship program focusing on practical machine learning applications in education._
