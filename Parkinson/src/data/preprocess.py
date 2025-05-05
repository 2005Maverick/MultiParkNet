import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load data from the specified file path
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data for model training
    """
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    data_path = "../../data/raw/parkinsons.csv"
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df) 