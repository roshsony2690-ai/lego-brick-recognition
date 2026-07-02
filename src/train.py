import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau
)
from data_preprocessing import load_data
from model import build_model

# Configuration
MODEL_SAVE_PATH = 'models/lego_model.h5'
EPOCHS = 50

def plot_training_history(history):
    """Plot training history to check for overfitting"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Loss
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history.history:
        ax3.plot(history.history['lr'], linewidth=2)
        ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Overfitting indicator (accuracy gap)
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    gap = [t - v for t, v in zip(train_acc, val_acc)]
    ax4.plot(gap, linewidth=2, color='red')
    ax4.axhline(y=0.1, color='orange', linestyle='--', label='Warning')
    ax4.axhline(y=0.2, color='red', linestyle='--', label='Overfitting')
    ax4.set_title('Overfitting Indicator (Train-Val Gap)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Gap')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Training history saved as 'training_history.png'")

def train_model():
    """Main training function"""
    print("🚀 Starting LEGO Brick Recognition Training...")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load data
    train_gen, val_gen = load_data()
    
    if train_gen is None:
        print("❌ Failed to load data!")
        return None, None
    
    num_classes = len(train_gen.class_indices)
    print(f"\n📊 Training for {num_classes} LEGO brick classes")
    
    # Build model
    model = build_model(num_classes, use_transfer_learning=True)
    
    # Callbacks to prevent overfitting
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n🔄 Starting training...")
    print("-" * 50)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n📊 Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"📊 Final Validation Loss: {val_loss:.4f}")
    
    # Check for overfitting
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc
    
    if gap > 0.15:
        print(f"\n⚠️ Warning: Model might be overfitting!")
        print(f"   Training Accuracy: {final_train_acc*100:.2f}%")
        print(f"   Validation Accuracy: {final_val_acc*100:.2f}%")
        print(f"   Gap: {gap*100:.2f}%")
        print("   Consider: adding more data, increasing dropout, or reducing model complexity")
    else:
        print(f"\n✅ Model seems well-regularized!")
        print(f"   Training Accuracy: {final_train_acc*100:.2f}%")
        print(f"   Validation Accuracy: {final_val_acc*100:.2f}%")
        print(f"   Gap: {gap*100:.2f}%")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()