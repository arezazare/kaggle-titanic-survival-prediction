# Titanic Survival Prediction – Kaggle Dataset

## 🚢 Overview
This project analyzes the Titanic dataset from Kaggle to predict passenger survival based on various features such as age, gender, ticket class, and family relationships. The goal is to build a machine learning model that accurately classifies whether a passenger survived or not.

## 📊 Dataset
- Source: [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- Rows: 891 (training set), 418 (test set)
- Features: Name, Age, Sex, Pclass, SibSp, Parch, Fare, Embarked, Cabin, Ticket

## 🛠️ Technologies & Tools
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebook
- Git / GitHub
- Optional: XGBoost, LightGBM, Hyperparameter Tuning

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

## ✅ Results
- Best Model: Random Forest Classifier
- Accuracy: ~82% on validation set
- Feature Importance: Sex, Fare, Pclass, Title

## 📁 Files
- `titanic_analysis.ipynb` – EDA and preprocessing
- `titanic_modeling.ipynb` – Model training and evaluation
- `submission.csv` – Final prediction file
- `plots/` – Folder containing key EDA and feature importance visualizations

## 🚀 How to Run
1. Clone the repository  
2. Open `titanic_modeling.ipynb` in Jupyter Notebook  
3. Run all cells step-by-step  
4. Optionally generate a new submission file

## 👏 Author
**Reza** — Full ML pipeline implemented from scratch.

## 📎 Resources
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Titanic Data Dictionary](https://www.kaggle.com/code/sashankpillai/titanic-data-dictionary)

## 📎 Credits
Dataset sourced from [Kaggle](https://www.kaggle.com/) for educational and research purposes. All rights to the original dataset belong to the respective uploader
