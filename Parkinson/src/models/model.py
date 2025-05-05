import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape):
    """
    Create the neural network model architecture
    """
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, filepath):
    """
    Save the trained model
    """
    model.save(filepath)

def load_model(filepath):
    """
    Load a saved model
    """
    return models.load_model(filepath) 