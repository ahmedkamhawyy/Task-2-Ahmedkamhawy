# Task-2-Ahmedkamhawy
# Supervised Learning Engine: KNN Data Classifier

A predictive AI classification model demonstrating the foundational pipeline of Supervised Learning. This project marks the architectural shift from heuristic, hardcoded rule logic to machine-derived pattern recognition. 

By providing historical data, this engine empowers the machine to derive its own logic and make intelligent decisions with high precision.

## 🧠 Architectural Overview

This system is built strictly on the Input-Process-Output (IPO) framework, designed to safely ingest raw data, process it algorithmically, and validate the output rigorously.

### The Pipeline
1. **INPUT (The Dataset):** * Ingests 150 balanced samples across 3 distinct classes (Setosa, Versicolor, Virginica).
   * Analyzes 4 dimensions: Sepal Length, Sepal Width, Petal Length, and Petal Width.
2. **PROCESS (The Logic Skeleton):**
   * **Data Split:** Randomizes and shuffles data, splitting it into an 80% Training Set (for pattern recognition) and a 20% Test Set (for validation).
   * **Scaling:** Applies `StandardScaler` (Mean = 0, Variance = 1) to eliminate dimensional bias.
   * **The Algorithm:** Utilizes the K-Nearest Neighbors (KNN) algorithm, tuned to an optimal value of K=5.
3. **OUTPUT (Validation):**
   * Computes the **Confusion Matrix** to track True Positives, False Positives, True Negatives, and False Negatives.
   * Calculates the **F1 Score** (the harmonic mean of Precision and Recall) to ensure strategic trustworthiness and sensitivity.

## 🛠️ Tech Stack & Workflow

* **Language:** Python 3.x
* **Libraries:** `scikit-learn`, `numpy`
* **Workflow Elements:**
  * `Instantiate`: Build the frame (`KNeighborsClassifier`).
  * `Fit`: Memorize the map (`model.fit`).
  * `Predict`: Apply logic (`model.predict`).

## 💻 Code Structure

```text
├── classifier_cli.py     # Main execution file containing the ML IPO pipeline
└── README.md             # Project documentation
