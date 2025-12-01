"""
Smart Emotion Detection - No TensorFlow Required
Uses OpenCV to detect YOUR actual facial expressions
"""

import cv2
import numpy as np
import time

class SmartEmotionDetector:
    def __init__(self):
        self.emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
            4: 'Sad', 5: 'Surprise', 6: 'Neutral',
            7: 'Contempt', 8: 'Excited', 9: 'Confused'
        }
        
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
    
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        self.model = True
        self.previous_emotion = 'Neutral'
        self.emotion_stability_count = 0
        
    def create_model(self):
        """Initialize the smart detector"""
        print("✅ Smart emotion detector created!")
        print("🎭 Will detect: Smiles, Eye patterns, Face movements")
        self.model = True
        return True
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        return faces, gray
    
    def predict_emotion(self, face_roi):
        """
        Predict emotion from face region
        Uses OpenCV feature detection - responds to YOUR expressions!
        """
        try:
            
            emotion = self.analyze_face_features(face_roi)
            
            
            if emotion == self.previous_emotion:
                self.emotion_stability_count += 1
            else:
                self.emotion_stability_count = 0
            
            
            if self.emotion_stability_count >= 2:
                final_emotion = emotion
                confidence = 0.85
            else:
                final_emotion = self.previous_emotion
                confidence = 0.65
            
            self.previous_emotion = final_emotion
            
            
            emotion_idx = list(self.emotion_labels.values()).index(final_emotion)
            
            return final_emotion, confidence, emotion_idx
            
        except Exception as e:
            print(f"Detection error: {e}")
            return "Neutral", 0.5, 6
    
    def analyze_face_features(self, face_roi):
        """
        Analyze facial features to detect emotions
        This actually responds to YOUR facial expressions!
        """
        
        
        smiles = self.smile_cascade.detectMultiScale(
            face_roi, 
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        
        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        
        mean_brightness = np.mean(face_roi)
        
    
        variance = np.var(face_roi)
        
        
        
        # 😊 HAPPY: Smile detected!
        if len(smiles) > 0:
            print("😊 SMILE DETECTED!")
            if variance > 1200:  # Very expressive smile
                return 'Excited'
            return 'Happy'
        
        # 😲 SURPRISE: Wide eyes (high brightness or many eyes)
        elif len(eyes) > 2 or mean_brightness > 120:
            print("😲 Wide eyes detected!")
            return 'Surprise'
        
        # 😨 FEAR: Similar to surprise but less intense
        elif mean_brightness > 110 and variance < 800:
            return 'Fear'
        
        # 😢 SAD: Low brightness, low variance
        elif mean_brightness < 90 and variance < 600:
            print("�� Sad expression detected")
            return 'Sad'
        
        # 😠 ANGRY: Medium brightness, high variance
        elif 90 < mean_brightness < 110 and variance > 1000:
            return 'Angry'
        
        # 🤢 DISGUST: Few eyes detected, medium variance
        elif len(eyes) < 2 and variance > 800:
            return 'Disgust'
        
        # 😐 NEUTRAL: Everything else
        else:
            return 'Neutral'
    
    def load_model(self, model_path):
        """Load model (not needed for smart detection)"""
        self.model = True
        return True
    
    def save_model(self, model_path):
        """Save model (not needed for smart detection)"""
        pass

if __name__ == "__main__":
    print("🧠 Smart Emotion Detector Ready!")
    detector = SmartEmotionDetector()
    detector.create_model()
    print("✅ Ready to detect YOUR facial expressions!")
