from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

class CancerModel:
    def __init__(self):
        self.model = LogisticRegression()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, precision, recall, conf_matrix
    
    def save_model(self, model_path, scaler_path):
        joblib.dump(self.model, model_path)