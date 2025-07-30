# Predictive Modelling of Scaling Using Sensor Data

This project implements a logistic regression-based model to **predict industrial scaling events** using time-series sensor data. The aim is to proactively identify high-risk situations and enable better maintenance decisions.

## 🧠 Overview

- **Data Source:** Sensor data stored in `test_orange.xlsx` with numerical features and a binary target variable `Scaling`.
- **Model Used:** Logistic Regression (with Scikit-learn)
- **Evaluation Metrics:** Accuracy Score, McFadden's pseudo R², Log Loss
- **Outcome:** Achieved high accuracy in predicting scaling events and extracted a readable logistic regression equation.

---

## 🚀 Features

- Automatic preprocessing (feature selection, standardization)
- Human-readable model interpretation (logistic equation)
- Quantitative evaluation with standard metrics
- McFadden’s pseudo R² for model goodness-of-fit
- Ready-to-use pipeline with Excel input

---

## 📊 Sample Output
Logistic Regression Equation:
logit(p) = -0.732 + (1.214 * Temperature) + (-0.943 * pH) + (0.556 * FlowRate) + (-0.385 * Pressure)

Probability(p) = 1 / (1 + exp(-logit(p)))

Model Accuracy: 86.21%
McFadden's pseudo R²: 0.3125
