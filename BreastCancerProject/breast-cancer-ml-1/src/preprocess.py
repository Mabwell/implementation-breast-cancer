def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    This function can be customized based on the dataset's requirements.
    """
    # Example: Fill missing values with the mean of the column
    for column in df.columns:
        if df[column].isnull().any():
            df[column].fillna(df[column].mean(), inplace=True)
    return df

def scale_features(X):
    """
    Scale features using StandardScaler.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def preprocess_data(df):
    """
    Preprocess the input DataFrame.
    This includes handling missing values and scaling features.
    """
    df = handle_missing_values(df)
    X = df.drop('target', axis=1)  # Assuming 'target' is the label column
    y = df['target']
    X_scaled, scaler = scale_features(X)
    return X_scaled, y, scaler