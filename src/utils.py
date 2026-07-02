# src/utils.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def get_class_names(dataset_path='dataset'):
    """Get sorted class names from dataset folder"""
    if not os.path.exists(dataset_path):
        return []
    return sorted([d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))])

def calibrate_predictions(predictions, temperature=1.5):
    """Apply temperature scaling to reduce overconfidence"""
    predictions = predictions + 1e-7
    scaled = np.exp(np.log(predictions) / temperature)
    scaled = scaled / np.sum(scaled, axis=-1, keepdims=True)
    return scaled

def preprocess_image(img_path, target_size=(128, 128)):  # Changed default to 128
    """Load and preprocess image for prediction"""
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


def get_top_predictions(predictions, class_names, top_k=3):
    """Get top K predictions with confidence scores"""
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'class': class_names[idx],
            'confidence': float(predictions[0][idx] * 100)  # Convert to Python float
        })
    return results