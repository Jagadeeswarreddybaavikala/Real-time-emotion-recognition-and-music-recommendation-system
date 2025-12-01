"""
Real-time Emotion Recognition Model
Optimized for Mac M2 with MPS (Metal Performance Shaders)
Supports 10 emotions with high accuracy
"""

import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import os

try:
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ Mac M2 GPU (MPS) acceleration enabled!")
    else:
        print("⚠️  Using CPU for training/inference")
except:
    print("⚠️  GPU configuration failed, using CPU")

class EmotionRecognitionModel:
    def __init__(self):
        self.img_size = 48
        self.num_classes = 10  
        
        
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust', 
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral',
            7: 'Contempt',    
            8: 'Excited',     
            9: 'Confused'     
        }
        
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def create_model(self):
        """Create an optimized CNN model for emotion recognition"""

        inputs = keras.Input(shape=(self.img_size, self.img_size, 1))
        
        
        x = layers.Rescaling(1./255)(inputs)
        
        
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
    
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_face(self, face_image):
        """Preprocess face image for emotion prediction"""
        
        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        
        face_image = cv2.resize(face_image, (self.img_size, self.img_size))
        
        
        face_image = face_image.astype('float32') / 255.0
        
    
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.expand_dims(face_image, axis=-1)
        
        return face_image
    
    def detect_faces(self, frame):
        """Detect faces in the frame using Haar Cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces, gray
    
    def predict_emotion(self, face_image):
        """Predict emotion from face image"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load or train the model first.")
        
        preprocessed_face = self.preprocess_face(face_image)
       
        prediction = self.model.predict(preprocessed_face, verbose=0)
        emotion_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return self.emotion_labels[emotion_idx], confidence, emotion_idx
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def save_model(self, model_path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(model_path)
        print(f"✅ Model saved successfully to {model_path}")

if __name__ == "__main__":

    emotion_model = EmotionRecognitionModel()
    model = emotion_model.create_model()
    print("✅ Model created successfully!")
    print(f"Model summary:")
    model.summary()
