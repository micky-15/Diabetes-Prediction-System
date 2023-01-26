import pandas as pd

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve


def prediction_model(): 
    df = pd.read_csv(
        "D:\\OneDrive\\DocumentsAndPrograms\\Python\\HealthCare\\diabetes.csv")
    # Evaluating algorithms


    def get_models():
        models = dict()

        # define the pipeline
        scaler = MinMaxScaler()
        power = PowerTransformer(method='yeo-johnson')

        clf1 = RandomForestClassifier()
        clf2 = CatBoostClassifier(verbose=False)
        clf3 = XGBClassifier()
        clf4 = LGBMClassifier()
        clf5 = LogisticRegression()

        models['Random Forest'] = Pipeline(
            steps=[('s', scaler), ('p', power), ('m', clf1)])
        models['Cat Boost'] = Pipeline(
            steps=[('s', scaler), ('p', power), ('m', clf2)])
        models['XGBoost'] = Pipeline(
            steps=[('s', scaler), ('p', power), ('m', clf3)])
        models['LightGBM'] = Pipeline(
            steps=[('s', scaler), ('p', power), ('m', clf4)])
        models['Logistic Regression'] = Pipeline(
            steps=[('s', scaler), ('p', power), ('m', clf5)])

        return models

    # evaluate a given model using cross-validation


    def evaluate_model(model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)
        scores = cross_val_score(
            model, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
        return scores


    # define dataset
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    y = y.values.ravel()

    # get the models to evaluate
    models = get_models()

    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        scores = scores
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))


    # define the pipeline
    model = RandomForestClassifier()

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)

    # Defining your search space
    hyperparameter_space = {
        "n_estimators": [25, 50, 75],
        "criterion": ["gini"],
        "max_depth": [3, 5, 10, None],
        "class_weight": ["balanced"],
        "min_samples_split": [0.001, 0.01, 0.05, 0.1],
    }

    global clf
    clf = GridSearchCV(model, hyperparameter_space,
                    scoring='f1_weighted', cv=cv,
                    n_jobs=-1, refit=True)


    # define dataset
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    y = y.values.ravel()

    # Run the GridSearchCV class
    clf.fit(X, y)

    # Print the best set of hyperparameters
    print(clf.best_params_, clf.best_score_)

    # Finalize Model
    clf = RandomForestClassifier(class_weight='balanced',
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=0.001,
                                n_estimators=75)
    # define dataset
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    y = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)

    clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)
    # y_pred_prob = clf.predict_proba(X_test)

    # importances = clf.feature_importances_
   
    # forest_importances = pd.Series(importances, index=X.columns)

    # target_names = ['Control', 'Patient']
    # print(classification_report(y_test, y_pred, target_names=target_names))
def predict(x):
    return clf.predict(x)
