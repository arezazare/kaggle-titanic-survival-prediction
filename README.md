# Titanic Survival Prediction – Kaggle Competition

## 🚢 Overview
This project tackles the classic Titanic survival classification challenge hosted on Kaggle.
The goal is to predict whether a passenger survived based on features like age, sex, fare, class, and family information.

A full machine learning pipeline was implemented:
- ✅ Data cleaning, feature engineering, and preprocessing
- ✅ Training with multiple models and hyperparameter tuning (GridSearchCV)
- ✅ 5-Fold Cross-validation and final model deployment
- 🎯 Final Kaggle Score: 0.77033 accuracy

## 📊 Dataset
- Source: Kaggle Titanic Dataset
- Training Set: 891 rows × 12 columns
- Test Set: 418 rows × 11 columns
- Features: Name, Age, Sex, Pclass, SibSp, Parch, Fare, Embarked, Cabin, Ticket
- Target: Survived (0 = did not survive, 1 = survived)

## 🧪 Key Tasks
- Data exploration and missing value analysis
- Feature engineering: extracting titles, family size, simplified deck, fare bands, child flag
- Handling categorical variables via One-Hot Encoding
- Feature scaling with MinMaxScaler
- Training and evaluating multiple models
- Hyperparameter tuning with GridSearchCV
- Model validation with 5-Fold Cross-Validation
- Preparing final Kaggle submission

## 🔍 Key Steps
1. **Data Cleaning:**
   - Handling missing values (e.g., Age, Embarked)
   - Converting categorical to numeric (e.g., Sex, Embarked)
   - Feature engineering (e.g., Title extraction from Name)

2. **Exploratory Data Analysis (EDA):**
   - Visualization of survival rates across features
   - Correlation heatmap
   - Distribution plots and boxplots

3. **Modeling:**
   - Logistic Regression
   - Random Forest
   - (Optional: SVM, Gradient Boosting, or Voting Classifier)

4. **Evaluation:**
   - Accuracy, Precision, Recall, F1-Score
   - Cross-validation
   - Confusion Matrix

5. **Submission:**
   - Prepared CSV file for Kaggle submission format: `PassengerId, Survived`

## 🔧 Technologies Used
- Python (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM)
- Jupyter Notebook
- Kaggle Kernels & APIs
- Git & GitHub
- Joblib (for saving final model and scaler)

## 🧠 What I Learned
- How careful feature engineering can drive model performance
- The importance of cross-validation for small datasets
- How to tune hyperparameters systematically using GridSearchCV
- Full machine learning pipeline from raw data to Kaggle submission
- Deploying trained models and scalers for reproducibility
- 
## 👏 Author
**Reza** — Full ML pipeline implemented from scratch.

## 📎 Resources
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Titanic Data Dictionary](https://www.kaggle.com/code/sashankpillai/titanic-data-dictionary)

## 📎 Credits
Dataset sourced from [Kaggle](https://www.kaggle.com/) for educational and research purposes. All rights to the original dataset belong to the respective uploader
