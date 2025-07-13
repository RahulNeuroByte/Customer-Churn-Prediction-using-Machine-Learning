# tuning.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# 1️ Logistic Regression
def tune_logistic_regression(X_train, y_train):
    print("Tuning Logistic Regression...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2']
    }

    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("Best Logistic Regression Params:", grid.best_params_)
    return grid.best_estimator_


# 2️ Decision Tree
def tune_decision_tree(X_train, y_train):
    print("Tuning Decision Tree...")
    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("Best Decision Tree Params:", grid.best_params_)
    return grid.best_estimator_


# 3️ Random Forest (RandomizedSearch + GridSearch)
def tune_random_forest(X_train, y_train):
    print("RandomizedSearchCV for Random Forest...")
    random_params = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 10, 20, 30, 40, None],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4]
    }

    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=random_params,
        n_iter=10,
        scoring='f1',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print("RandomizedSearch Best Params:", best_params)

    grid_params = {
        'n_estimators': [best_params['n_estimators']],
        'max_depth': [best_params['max_depth']],
        'min_samples_split': [best_params['min_samples_split']],
        'min_samples_leaf': [best_params['min_samples_leaf']]
    }

    print("GridSearchCV for Random Forest (Refinement)...")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=grid_params,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    print("Final Random Forest Params:", grid_search.best_params_)
    return grid_search.best_estimator_


# 4️ Support Vector Machine (SVM)
def tune_svm(X_train, y_train):
    print("Tuning SVM...")
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("Best SVM Params:", grid.best_params_)
    return grid.best_estimator_


# 5️ K-Nearest Neighbors (KNN)
def tune_knn(X_train, y_train):
    print("🔍 Tuning KNN...")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("Best KNN Params:", grid.best_params_)
    return grid.best_estimator_


# 6️ Naive Bayes (Minimal tuning)
def tune_naive_bayes(X_train, y_train):
    print("Tuning Naive Bayes (minimal tuning)...")
    # Not many parameters to tune; just returning model
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Naive Bayes model trained (default settings).")
    return model
