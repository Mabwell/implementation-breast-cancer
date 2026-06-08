import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def extract_features(X, y, k=10):
    """
    Extract the top k features from the dataset using ANOVA F-value.
    
    Parameters:
    X (pd.DataFrame): Feature dataset.
    y (pd.Series): Target variable.
    k (int): Number of top features to select.
    
    Returns:
    pd.DataFrame: DataFrame containing the selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    
    return X.iloc[:, selected_features]

def feature_importance(model, X, feature_names):
    """
    Calculate and return feature importance from the trained model.
    
    Parameters:
    model: Trained machine learning model.
    X (pd.DataFrame): Feature dataset.
    feature_names (list): List of feature names.
    
    Returns:
    pd.DataFrame: DataFrame containing feature names and their importance scores.
    """
    importance = model.coef_[0] if hasattr(model, 'coef_') else model.feature_importances_
    return pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)