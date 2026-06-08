import os
import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.train_model import train_model, evaluate_model

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        data = load_breast_cancer()
        self.X = data.data
        target_names = data.target_names
        malignant_idx = int(np.where(target_names == 'malignant')[0][0])
        self.y = np.where(data.target == malignant_idx, 1, 0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

    def test_train_model(self):
        model, scaler = train_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)

    def test_evaluate_model(self):
        model, scaler = train_model(self.X_train, self.y_train)
        X_test_scaled = scaler.transform(self.X_test)
        y_pred = model.predict(X_test_scaled)
        accuracy = evaluate_model(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.5)  # Expecting accuracy to be greater than 50%

if __name__ == '__main__':
    unittest.main()