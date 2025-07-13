# 📊 Telco Customer Churn Prediction - Machine Learning Pipeline

This project demonstrates a complete end-to-end machine learning workflow using the **Telco Customer Churn Dataset**. The objective is to build and compare multiple classification models to predict whether a customer is likely to churn, and identify the best-performing model using evaluation metrics and hyperparameter tuning.

---

## 📁 Dataset Used

**Dataset:** Telco Customer Churn  
- Applied data cleaning and preprocessing  
- Removed irrelevant features  
- Performed encoding and scaling  
- Conducted exploratory data analysis and visualization  
- Trained and evaluated multiple ML models  
- Applied hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`

---

## 🎯 Objective

- Train and compare multiple machine learning models
- Evaluate models using Accuracy, Precision, Recall, and F1-Score
- Tune models using hyperparameter search techniques
- Analyze and select the best-performing model based on business objectives

---

## 🤖 Models Trained

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

---

## 📈 Visualizations Created

- Class distribution of the target variable
- Correlation heatmap
- Before vs After Scaling comparison
- Accuracy, Precision, Recall, F1-Score comparison (before and after tuning)

---

## 📊 Evaluation Metrics - Before Tuning

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.8038   | 0.6485    | 0.5722 | 0.6080   |
| Decision Tree      | 0.7050   | 0.4459    | 0.4519 | 0.4489   |
| Random Forest      | 0.7797   | 0.6096    | 0.4759 | 0.5345   |
| SVM                | 0.7910   | 0.6351    | 0.5027 | 0.5612   |
| KNN                | 0.7527   | 0.5359    | 0.5187 | 0.5272   |
| Naive Bayes        | 0.7356   | 0.5018    | 0.7513 | 0.6017   |

---

## ⚙️ Evaluation Metrics - After Hyperparameter Tuning

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.8038   | 0.6485    | 0.5722 | 0.6080   |
| Decision Tree      | 0.7754   | 0.5980    | 0.4733 | 0.5284   |
| Random Forest      | 0.7910   | 0.6408    | 0.4866 | 0.5532   |
| SVM                | 0.7989   | 0.6409    | 0.5535 | 0.5940   |
| KNN                | 0.7719   | 0.5730    | 0.5561 | 0.5645   |
| Naive Bayes        | 0.7356   | 0.5018    | 0.7513 | 0.6017   |

---

## 🏆 Best Performing Model

- **Logistic Regression**: Highest accuracy and balanced performance even after tuning. Interpretable and generalizes well.
- **Naive Bayes**: Highest recall — useful when the goal is to catch more churners (important for retention strategies).
- **SVM & Random Forest**: Achieved strong balance between precision and recall after tuning.

---

## ✅ Conclusion

This project successfully covered the entire machine learning pipeline:

- 📌 **Preprocessing:** Feature removal, encoding, and scaling  
- 🧠 **Modeling:** Trained six classifiers and compared their performance  
- ⚙️ **Tuning:** Improved results using GridSearchCV & RandomizedSearchCV  
- 📉 **Evaluation:** Focused on meaningful metrics like recall and F1-score for imbalanced data  
- 📊 **Visualization:** Enhanced interpretability and presentation of results  

### Key Takeaways:

- 📌 No single model fits all — selection depends on business goals (maximize **recall** vs **precision**)
- 📌 **Naive Bayes** is ideal when false negatives are costlier
- 📌 **Logistic Regression** is a solid, reliable baseline
- 📌 Hyperparameter tuning plays a vital role in boosting model performance

---

## 📂 Project Structure

