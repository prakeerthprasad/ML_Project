# src/baseline_models.py

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)


def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, min_samples_split=20,random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "name": "DecisionTree",
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred, output_dict=False),
        "cm": confusion_matrix(y_test, y_pred)
    }
    return metrics, y_pred


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "name": "RandomForest",
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred, output_dict=False),
        "cm": confusion_matrix(y_test, y_pred)
    }
    return metrics, y_pred
