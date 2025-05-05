import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ..data.preprocess import load_data, preprocess_data
from .model import load_model

def evaluate_model(model_path, data_path):
    """
    Evaluate the trained model on test data
    """
    # Load data and model
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = load_model(model_path)
    
    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../../models/confusion_matrix.png')
    plt.close()
    
    return report, cm

if __name__ == "__main__":
    # Example usage
    model_path = "../../models/parkinsons_model.h5"
    data_path = "../../data/raw/parkinsons.csv"
    
    report, cm = evaluate_model(model_path, data_path)
    print("Classification Report:")
    print(report) 