import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
from data_loader import load_data
from preprocess import preprocess_data

def train_model():
    # Load the dataset
    data = load_data('data/raw/breast_cancer_data.csv')
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    
    # Save the trained model and scaler
    joblib.dump(model, 'models/cancer_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

if __name__ == '__main__':
    train_model()