import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from models.efficientnet import CropDiseaseModel
from data.dataset import PlantVillageDataset

def train_model(config):
    """Train the crop disease prediction model."""
    print("Loading dataset...")
    dataset = PlantVillageDataset(
        data_dir=config.DATASET_PATH,
        img_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )
    
    train_ds, val_ds, test_ds = dataset.load_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT
    )
    
    print("Building model...")
    model = CropDiseaseModel(
        num_classes=config.NUM_CLASSES,
        img_size=config.IMG_SIZE
    )
    
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("Training model...")
    history = model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )
    
    plot_training_history(history)
    
    print("Evaluating model...")
    test_loss, test_acc = model.model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return model, history

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
   
    os.makedirs('training/plots', exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training/plots/training_history.png')
    plt.close()

if __name__ == "__main__":
    from config import DevelopmentConfig
    
    train_model(DevelopmentConfig)