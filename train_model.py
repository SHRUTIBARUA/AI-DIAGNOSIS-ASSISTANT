import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths for training data (Ensure these folders have images)
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_SAVE_PATH = "models/diagnostic_model.h5"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Define a simple CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (Positive/Negative)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
model = create_model()

# Data augmentation for training images
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load images from dataset (Ensure dataset exists)
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(224,224), batch_size=32, class_mode='binary')

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=5)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"Model saved successfully at {MODEL_SAVE_PATH}")
