from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, 
    GlobalAveragePooling2D, BatchNormalization, Input
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

IMG_SIZE = (224, 224)

def build_model(num_classes, use_transfer_learning=True):
    """
    Build CNN model with strong regularization to prevent overfitting
    """
    
    if use_transfer_learning:
        # Transfer Learning with ResNet50
        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=(*IMG_SIZE, 3)
        )
        base_model.trainable = False  # Freeze base layers
        
        model = Sequential([
            Input(shape=(*IMG_SIZE, 3)),
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
    else:
        # Custom CNN with strong regularization
        model = Sequential([
            Input(shape=(*IMG_SIZE, 3)),
            
            # Block 1
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            
            # Block 2
            Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            
            # Block 3
            Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            
            # Block 4
            Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.4),
            
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
    
    # Use lower learning rate for stability
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model