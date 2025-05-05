import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from ..data.preprocess import load_data, preprocess_data
from .model import create_model, save_model

def train_model(data_path, model_save_path, epochs=100, batch_size=32):
    """
    Train the model on the provided data
    """
    # Load and preprocess data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Create and compile model
    model = create_model(input_shape=(X_train.shape[1],))
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_accuracy'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    return model, history

if __name__ == "__main__":
    # Example usage
    data_path = "../../data/raw/parkinsons.csv"
    model_save_path = "../../models/parkinsons_model.h5"
    
    model, history = train_model(data_path, model_save_path) 