"""
Module 5 Week A — Core Skills Drill: Classification & Evaluation Basics

Complete the three functions below. Each function has a docstring
describing its inputs, outputs, and purpose.

Run your work: python drill.py
Test your work: the autograder runs automatically when you open a PR.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(df, target_col="churned", test_size=0.2, random_state=42):
    """Split a DataFrame into train and test sets with stratification."""
    
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size:     {len(X_test)}")
    print(f"Train churn rate:  {y_train.mean():.3f}")
    print(f"Test churn rate:   {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics from true and predicted labels."""

    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred))
    }
    
    return metrics


def run_cross_validation(X_train, y_train, n_folds=5, random_state=42):
    """Run stratified k-fold cross-validation with LogisticRegression."""
    model = LogisticRegression(
        random_state=random_state, 
        max_iter=1000, 
        class_weight="balanced"
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=n_folds, scoring="accuracy")
    
    results = {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores))
    }
    
    return results


if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_csv("data/telecom_churn.csv")
        print(f"Loaded {len(df)} rows")
    except FileNotFoundError:
        print("Error: data/telecom_churn.csv not found. Please check your data directory.")
        exit()

    # Task 1: Split
    numeric_cols = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen", "has_partner",
                    "has_dependents"]
    df_numeric = df[numeric_cols + ["churned"]]

    result = split_data(df_numeric)
    if result is not None:
        X_train, X_test, y_train, y_test = result


        # Task 2: Metrics
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred)
        if metrics:
            print(f"Metrics: {metrics}")



        # Task 3: Cross-validation
        cv_results = run_cross_validation(X_train, y_train)
        if cv_results:
            print(f"CV: {cv_results['mean']:.3f} +/- {cv_results['std']:.3f}")