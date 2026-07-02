import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import check_dataset_balance, get_class_names

# Configuration
DATASET_PATH = 'dataset'
IMG_SIZE = (224, 224)  # Increased for better accuracy
BATCH_SIZE = 16  # Smaller batch for better generalization
VALIDATION_SPLIT = 0.2

def load_data():
    """Load and prepare data with augmentation"""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder '{DATASET_PATH}' not found!")
    
    # Check dataset balance
    check_dataset_balance(DATASET_PATH)
    
    # Training data with aggressive augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    
    # Validation data - only rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print(f"\n✅ Found {train_generator.samples} training images")
    print(f"✅ Found {val_generator.samples} validation images")
    print(f"📋 Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator

if __name__ == "__main__":
    # Test the data loading
    train_gen, val_gen = load_data()
    print("\n✅ Data loading successful!")
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")