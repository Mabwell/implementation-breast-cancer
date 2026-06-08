# Breast Cancer Machine Learning Project

This project aims to develop a machine learning model to predict breast cancer using the Breast Cancer Wisconsin dataset. The model is built using Logistic Regression and includes various preprocessing steps to ensure data quality and model performance.

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
├── requirements.txt          # Project dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd breast-cancer-ml
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.

   ```
   pip install -r requirements.txt
   ```

3. **Data Preparation:**
   Place the raw Breast Cancer Wisconsin dataset files in the `data/raw` directory. The dataset should be in a format compatible with the data loading functions.

4. **Run the Model Training:**
   Execute the `train_model.py` script to preprocess the data, train the model, and save the trained model and scaler.

   ```
   python src/train_model.py
   ```

5. **Explore the Data:**
   Use the Jupyter notebook located in the `notebooks` directory for exploratory data analysis and visualizations.

## Project Objectives

- Load and preprocess the Breast Cancer Wisconsin dataset.
- Train a Logistic Regression model to classify breast cancer cases.
- Evaluate the model's performance using metrics such as Accuracy, Precision, Recall, and Confusion Matrix.
- Provide a clear structure for further development and testing of the model.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.