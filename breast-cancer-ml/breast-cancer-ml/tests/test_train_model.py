import unittest
from src.train_model import train_model, evaluate_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Load the Breast Cancer dataset
        self.data = load_breast_cancer()
        self.X = self.data.data
        self.y = self.data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def test_train_model(self):
        model, scaler = train_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)

    def test_evaluate_model(self):
        model, scaler = train_model(self.X_train, self.y_train)
        predictions = model.predict(scaler.transform(self.X_test))
        accuracy = evaluate_model(self.y_test, predictions)
        self.assertGreaterEqual(accuracy, 0.9)  # Expecting at least 90% accuracy

if __name__ == '__main__':
    unittest.main()