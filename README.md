# Iris Flower Classification Model

A clean machine learning pipeline implementing the **K-Nearest Neighbors (KNN)** algorithm using **Scikit-Learn** and **Pandas** to classify Iris flower species based on floral measurements.

---

## 📊 Dataset Overview
The model utilizes the classic **Iris Dataset** (150 samples) to classify flowers into three species: `Setosa`, `Versicolor`, and `Virginica`. Predictions are made using four numeric attributes:
* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

---

## ⚡ Performance & Model Architecture
* **Framework:** Scikit-Learn (`sklearn`)
* **Model Classifier:** `KNeighborsClassifier(n_neighbors=3)`
* **Data Splitting:** 80% Training | 20% Testing (Stratified)
* **Model Accuracy:** `96.67%`

The model calculates vector proximity in the feature space using the mathematical standard **Euclidean Distance**:
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

---

## 🛠️ Project Structure
```text
├── Iris.csv          # Structured dataset
├── iris_model.py     # Scikit-learn machine learning pipeline
└── README.md         # Documentation

