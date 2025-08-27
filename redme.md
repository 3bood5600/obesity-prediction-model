# Obesity Risk Prediction

## Project Description
This project aims to build a classification model to predict the risk of obesity based on various lifestyle and physical attributes.

## Live Demo
A live Streamlit demo of the model is available here:
- https://3bood5600-obesity-prediction-model-app-usoa0s.streamlit.app/

## Data Source
The dataset used in this project is the Obesity Risk dataset, which can be found [https://www.kaggle.com/competitions/playground-series-s4e2]. It contains information about individuals including their[...]

## Steps Taken
1.  **Data Loading and Initial Exploration**: Loaded the dataset and performed initial checks for data types, missing values, and duplicates.
2.  **Exploratory Data Analysis (EDA)**: Visualized the distribution of various features and explored relationships between features and the target variable (Obesity Type).
3.  **Feature Engineering**: Created a new feature, BMI (Body Mass Index), based on height and weight.
4.  **Data Preprocessing**: Handled categorical features through encoding (Label Encoding and One-Hot Encoding) and assessed feature importance using Mutual Information and Cramer's V.
5.  **Splitting Data and Scaling**: Split the data into training and testing sets and applied scaling to numerical features. SMOTE was used to address potential class imbalance.
6.  **Model Training and Evaluation**: Trained and evaluated several classification models (Logistic Regression, Random Forest, and XGBoost) on the preprocessed data.
7.  **Hyperparameter Tuning and Cross-Validation**: Optimized the models using GridSearchCV and cross-validation to find the best hyperparameters.
8.  **Saving the Best Model**: The best performing model was saved for future use.

## How to Run
[Instructions on how to run the notebook or script, including any dependencies]

## Files
*   `train.csv`: The dataset used for training and evaluation.
*   `feature_names.pkl`: A file containing the names of the features used in the model.
*   `scaler.pkl`: A file containing the fitted scaler object.
*   `best_model.pkl`: The saved best performing model.
*   `README.md`: This file.


