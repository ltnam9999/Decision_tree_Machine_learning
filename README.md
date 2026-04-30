# Decision_tree_Machine_learning
This is a self-study project to improve my knowledge for my studies. I combined between summarize theory and vibe code
# 🌳 Decision Tree Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Focus-Data%20Science-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

This project focuses on implementing and evaluating **Decision Tree** models for both **Classification** and **Regression** tasks. It serves as a core component of my Data Science and Machine Learning research, demonstrating how tree-based algorithms handle diverse real-world datasets.

## 📂 Project Structure

All core logic and execution scripts are organized within the `Decision_tree_Project` directory for maximum modularity:
```text
Decision_tree_Project/
├── core/                # Core logic for Decision Tree implementation
├── data/                # Datasets (Heart, Drink, Teen Mental Health, etc.)
├── encoding/            # Data preprocessing and Label Encoding modules
├── modelpre/            # Model preprocessing utilities
├── results/             # Output directory for trees (.txt) and metrics (.json)
└── src/                 # Main execution scripts (Training & Evaluation)
```

## 🚀 Key Features

*   **Dual-Task Support**: Full implementation of Decision Trees for both **Classification** (categorical outcomes) and **Regression** (continuous values).
*   **Performance Analytics**: Automated calculation of essential evaluation metrics, including **Precision**, **Recall**, and **F1-score** to ensure model reliability.
*   **Multi-Dataset Testing**: The system is validated against diverse datasets such as Heart Disease, Drink Classification, and Teen Mental Health to guarantee robustness.
*   **Automated Export**: Comprehensive decision tree structures (.txt) and performance summaries (.json) are automatically exported to the `results/` folder for easy analysis.

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have Python installed (3.8+ recommended). You can install the necessary dependencies using pip:
pip install numpy pandas scikit-learn

### Running the Project

Since the execution scripts have been organized into the `src/` directory, use the following commands from the project root to run the models:

*   **To train and evaluate the Classification model:**
    ```bash
    python src/run_classifier.py
    ```

*   **To train and evaluate the Regression model:**
    ```bash
    python src/run_regressor.py
    ```

*   **To run evaluation separately:**
    ```bash
    python src/evaluate.py
    ```

## 📊 Datasets

The models in this project are trained and validated using high-quality datasets sourced from **Kaggle**. These datasets cover various domains, providing a robust testing ground for Decision Tree algorithms.

*   **Source**: All data files used in this repository were originally obtained from [Kaggle](https://www.kaggle.com/).
*   **Key Datasets**:
    *   **Heart Disease Dataset**: Used for binary classification to predict the presence of heart disease. [Link to Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
    *   **Student Mental Health**: Used to analyze and classify factors affecting the mental health of students. [Link to Dataset](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy)

> **Note**: For local execution, ensure these datasets are placed within the `Decision_tree_Project/data/` directory as structured in this repository.

## 📊 Evaluation Results
Model performance is logged in detail within `results/metrics_summary.json`. Current development focuses on optimizing tree depth and splitting criteria to maximize the F1-score across all test cases.

## 📄 Documentation
For detailed theoretical background and project reports, please refer to:
*   [Decision_tree.pdf](./Decision_tree.pdf)
*   [Decision_tree_eng.pdf](./Decision_tree_eng.pdf)

---
*Developed by **ltnam9999**. Feel free to explore the code and reach out for any discussions regarding Decision Tree algorithms!*
