# Breast Cancer Machine Learning Project

This project aims to develop a machine learning model to predict breast cancer using the Breast Cancer Wisconsin dataset. The model is built using Logistic Regression and includes preprocessing and evaluation steps.

## Project Structure

```
breast-cancer-ml
├── src
│   ├── train_model.py        # Script for training the model
│   ├── data_loader.py        # Functions to load the dataset
│   ├── preprocess.py         # Data preprocessing functions
│   ├── features.py           # Feature extraction and selection
│   ├── model.py              # Model implementation and evaluation
│   ├── utils.py              # Utility functions
│   └── __init__.py           # Package initialization
├── data
│   ├── raw                   # Directory for raw dataset files
│   └── processed             # Directory for processed datasets
├── models                    # Directory for saving trained models
├── notebooks
│   └── exploration.ipynb     # Jupyter notebook for exploratory data analysis
├── tests
│   ├── test_train_model.py   # Unit tests for training script
│   └── test_data_loader.py   # Unit tests for data loading
├── deployments               # Deployment artifacts (Docker, k8s, etc.)
├── requirements.txt          # Project dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd breast-cancer-ml
   ```

2. Install dependencies (recommended in a virtual environment):
   ```
   python -m pip install -r requirements.txt
   ```

3. Data Preparation:
   Place raw dataset files in `data/raw` (if using local files). The training script also supports using scikit-learn's built-in dataset.

4. Run training:
   ```
   python src/train_model.py
   ```
   Trained artifacts are saved to `models/`.

5. Run tests:
   ```
   python -m pytest -q
   ```

## Objectives

- Load and preprocess the Breast Cancer Wisconsin dataset.
- Train a Logistic Regression model to classify breast cancer cases.
- Evaluate the model using Accuracy, Precision, Recall, and Confusion Matrix.
- Provide a clear structure for development, testing, and deployment.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.