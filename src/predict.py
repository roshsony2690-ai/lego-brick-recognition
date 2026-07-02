import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import (
    get_class_names, 
    calibrate_predictions, 
    preprocess_image,
    get_top_predictions
)

# Configuration
MODEL_PATH = 'models/lego_model.h5'
DATASET_PATH = 'dataset'
IMG_SIZE = (224, 224)
TEMPERATURE = 1.5  # Adjust for confidence calibration

def predict_image(img_path, model_path=MODEL_PATH, temperature=TEMPERATURE):
    """
    Predict LEGO brick class from image
    With confidence calibration to prevent 100% predictions
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first!")
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, compile=False)
    
    # Get class names
    class_names = get_class_names(DATASET_PATH)
    if not class_names:
        raise ValueError("No classes found in dataset folder!")
    
    # Preprocess image
    print("Processing image...")
    img_array, original_img = preprocess_image(img_path, target_size=IMG_SIZE)
    
    # Predict
    print("Making prediction...")
    raw_predictions = model.predict(img_array, verbose=0)
    
    # Calibrate predictions (prevents overconfidence)
    calibrated_predictions = calibrate_predictions(raw_predictions, temperature)
    
    # Get results
    confidence = np.max(calibrated_predictions) * 100
    predicted_class = class_names[np.argmax(calibrated_predictions)]
    top_predictions = get_top_predictions(calibrated_predictions, class_names, top_k=3)
    
    # Display results
    print("\n" + "="*50)
    print("🔍 PREDICTION RESULTS")
    print("="*50)
    
    if confidence > 70:
        print(f"✅ Predicted Brick: {predicted_class}")
        print(f"   Confidence: {confidence:.2f}%")
    elif confidence > 50:
        print(f"⚠️ Uncertain Prediction: {predicted_class}")
        print(f"   Confidence: {confidence:.2f}%")
        print("   Try a clearer image or different angle")
    else:
        print("❌ This does not appear to be a LEGO brick")
        print(f"   Max confidence: {confidence:.1f}%")
    
    print("\n🏆 Top Predictions:")
    for i, pred in enumerate(top_predictions, 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.1f}%")
    
    # Visualize
    visualize_prediction(original_img, predicted_class, confidence, top_predictions)
    
    return predicted_class, confidence

def visualize_prediction(img, predicted_class, confidence, top_predictions):
    """Visualize prediction results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f"Predicted: {predicted_class}\nConfidence: {confidence:.1f}%", 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Top predictions bar chart
    names = [p['class'] for p in top_predictions]
    confs = [p['confidence'] for p in top_predictions]
    colors = ['green' if c > 50 else 'orange' if c > 30 else 'red' for c in confs]
    
    axes[1].barh(names, confs, color=colors)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Top Predictions')
    axes[1].set_xlim([0, 100])
    
    # Confidence distribution
    axes[2].hist([p['confidence'] for p in top_predictions], bins=10, alpha=0.7, color='blue')
    axes[2].set_xlabel('Confidence Score (%)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Confidence Distribution')
    axes[2].axvline(x=50, color='red', linestyle='--', label='Threshold')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n📊 Visualization saved as 'prediction_result.png'")

if __name__ == "__main__":
    # Test prediction
    test_image = 'path/to/test/lego/image.jpg'  # Change this
    if os.path.exists(test_image):
        predict_image(test_image)
    else:
        print(f"Test image not found at: {test_image}")
        print("Please update the path or test with a different image")