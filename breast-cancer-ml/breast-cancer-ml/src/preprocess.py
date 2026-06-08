def handle_missing_values(df):
    # Fill missing values with the mean of the column
    return df.fillna(df.mean())

def encode_labels(df, label_column):
    # Convert categorical labels to numerical values
    df[label_column] = df[label_column].astype('category').cat.codes
    return df

def preprocess_data(df, label_column):
    df = handle_missing_values(df)
    df = encode_labels(df, label_column)
    return df