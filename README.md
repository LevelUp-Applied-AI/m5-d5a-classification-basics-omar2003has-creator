# Telecom Customer Churn Prediction

##  Project Overview
This project focuses on building a classification pipeline to predict **Customer Churn**. The goal is to identify customers likely to leave the service based on their usage patterns and account information, allowing the company to take proactive retention measures.

##  The Approach (Implementation Details)

### 1. Data Engineering & Stratified Splitting
* **Feature Selection:** Filtered the dataset to focus on numerical attributes (such as `tenure`, `monthly_charges`, and `total_charges`) to ensure mathematical compatibility with the model.
* **Balanced Partitioning:** Used **Stratified Splitting** (80% Train / 20% Test) to ensure that both sets maintain the same ratio of "Churn" vs "No-Churn" as the original dataset, preventing biased training.

### 2. Model Training & Evaluation
* **Algorithm:** Implemented **Logistic Regression** with the `class_weight="balanced"` parameter. This is crucial for handling imbalanced data, as it gives more weight to the minority class (customers who churned).
* **Evaluation Metrics:** The model was evaluated using a comprehensive suite of metrics:
    * **Accuracy:** To measure overall correctness.
    * **Recall:** To ensure we capture as many actual churners as possible.
    * **Precision:** To measure the reliability of the churn predictions.
    * **F1-Score:** To find the harmonic mean between Precision and Recall.

### 3. Model Reliability (Cross-Validation)
* **K-Fold Validation:** To ensure the results weren't just a "lucky split," we performed **5-Fold Stratified Cross-Validation**.
* **Stability Check:** Calculated the **Mean** and **Standard Deviation** of the scores. A low standard deviation confirms that the model is stable and performs consistently across different data segments.

##  Results Summary
Based on the latest execution:
* **Accuracy:** ~63.7%
* **Recall:** ~51% (Successfully identifies over half of the potential churners).
* **Model Stability:** The very low standard deviation (±0.022) proves the model is reliable and not prone to overfitting.

