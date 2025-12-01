from spotify_music import play_music_for_emotion
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
sys.path.insert(0, 'src')
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import queue
import time
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False
from music_system import MusicRecommendationSystem
class TrainedEmotionDetector:
    """Uses YOUR trained model for accurate detection"""
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion_map = {
            'angry': 'Angry',
            'disgust': 'Disgust', 
            'fear': 'Fear',
            'happy': 'Happy',
            'neutral': 'Neutral',
            'sad': 'Sad',
            'surprise': 'Surprise'
        }    
        self.extended = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 
                        'Surprise', 'Neutral', 'Contempt', 'Excited', 'Confused']       
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )      
        self.model = None
        self.prev_emotion = 'Neutral'
        self.confidence_threshold = 0.5       
    def load_model(self):
        """Load the trained model"""
        model_path = 'models/accurate_emotion_model.h5'      
        if not os.path.exists(model_path):
            print(f"⚠️  Trained model not found at {model_path}")
            print("📝 Please run: python3.11 train_fer2013_model.py")
            return False   
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ TRAINED model loaded successfully!")
            print(f"🎯 Ready to detect YOUR emotions accurately!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False  
    def detect_faces(self, frame):
        """Detect faces"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        return faces, gray
    def predict_emotion(self, face_roi):
        """Predict emotion using TRAINED MODEL"""
        if self.model is None:
            return 'Neutral', 0.0  
        try:   
            face = cv2.resize(face_roi, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1) 
            predictions = self.model.predict(face, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[emotion_idx])
            if confidence > self.confidence_threshold:
                emotion = self.emotions[emotion_idx]
                emotion_name = self.emotion_map[emotion]
                emotion_name = self.extend_emotion(emotion_name, confidence, predictions)
                self.prev_emotion = emotion_name
                return emotion_name, confidence
            else:
                return self.prev_emotion, confidence 
        except Exception as e:
            print(f"Prediction error: {e}")
            return 'Neutral', 0.0
    def extend_emotion(self, base_emotion, confidence, predictions):
        """Extend 7 emotions to 10 based on confidence patterns"""
        if base_emotion == 'Happy' and confidence > 0.85:
            return 'Excited'
        elif base_emotion == 'Happy' and predictions[self.emotions.index('surprise')] > 0.2:
            return 'Excited'
        elif base_emotion == 'Surprise' and confidence < 0.6:
            return 'Confused'
        elif base_emotion == 'Angry' and predictions[self.emotions.index('disgust')] > 0.15:
            return 'Contempt'
        elif confidence < 0.55 and np.max(predictions) < 0.6:
            return 'Confused'
        return base_emotion
class AccurateEmotionApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("🎯 ACCURATE Emotion Detection + Music")
        self.root.geometry("1400x900")  
        self.detector = TrainedEmotionDetector()
        self.music = MusicRecommendationSystem()
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            print("Install with: python3.11 -m pip install tensorflow==2.13.0")
            return
        if not self.detector.load_model():
            print("❌ Cannot load trained model")
            print("Train first: python3.11 train_fer2013_model.py")
            return
        self.cap = None
        self.active = False
        self.emotion = "Neutral"
        self.confidence = 0.0
        self.frame_queue = queue.Queue(maxsize=2)
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
    def setup_ui(self):
        """Setup UI"""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        left = ctk.CTkFrame(self.root)
        left.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        title = ctk.CTkLabel(left, text="🎯 Emotion Recognition", 
                            font=("Arial", 20, "bold"))
        title.pack(pady=10)
        self.cam_btn = ctk.CTkButton(
            left, text="�� Start Camera",
            command=self.toggle, height=50, font=("Arial", 16)
        )
        self.cam_btn.pack(pady=10)
        self.video = ctk.CTkLabel(left, text="Camera feed")
        self.video.pack(expand=True, fill="both", padx=10, pady=10)
        self.emotion_lbl = ctk.CTkLabel(
            left, text="😐 Neutral (0%)",
            font=("Arial", 28, "bold")
        )
        self.emotion_lbl.pack(pady=10)
        right = ctk.CTkFrame(self.root)
        right.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(right, text="🎵 Music Recommendations", 
                    font=("Arial", 20, "bold")).pack(pady=20)
        self.songs_frame = ctk.CTkScrollableFrame(right)
        self.songs_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.update_songs()
    def toggle(self):
        """Toggle camera"""
        if not self.active:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.active = True
                self.cam_btn.configure(text="⏹️ Stop Camera")
                threading.Thread(target=self.camera_loop, daemon=True).start()
                threading.Thread(target=self.detect_loop, daemon=True).start()
                print("✅ Camera started - Show your expressions!")
        else:
            self.active = False
            if self.cap:
                self.cap.release()
            self.cam_btn.configure(text="📷 Start Camera")
    def camera_loop(self):
        """Camera capture loop"""
        while self.active:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                self.show_frame(frame)
            time.sleep(1/30)
    def detect_loop(self):
        """Emotion detection loop"""
        while self.active:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                faces, gray = self.detector.detect_faces(frame)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face = gray[y:y+h, x:x+w]
                    emotion, conf = self.detector.predict_emotion(face)
                    # Spotify playback integration
                    play_music_for_emotion(emotion)
                    if emotion != self.emotion or abs(conf - self.confidence) > 0.1:
                        print(f"🎭 {emotion} ({conf:.0%})")
                        self.emotion = emotion
                        self.confidence = conf
                        self.root.after(0, self.update_ui)
            time.sleep(0.1)
    def show_frame(self, frame):
        """Display frame"""
        faces, _ = self.detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{self.emotion} ({self.confidence:.0%})"
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.root.after(0, lambda: self.video.configure(image=img, text=""))
        self.root.after(0, lambda: setattr(self.video, 'image', img))
    def update_ui(self):
        """Update emotion display"""
        emojis = {
            'Happy': '😊', 'Sad': '😢', 'Angry': '😠', 'Fear': '😨',
            'Surprise': '😲', 'Disgust': '🤢', 'Neutral': '😐',
            'Contempt': '😤', 'Excited': '🤩', 'Confused': '😕'
        }
        emoji = emojis.get(self.emotion, '😐')
        self.emotion_lbl.configure(
            text=f"{emoji} {self.emotion} ({self.confidence:.0%})"
        )
        self.update_songs()
    def update_songs(self):
        """Update song list"""
        for w in self.songs_frame.winfo_children():
            w.destroy()
        songs = self.music.get_all_songs_for_emotion(self.emotion)
        for i, song in enumerate(songs[:10], 1):
            f = ctk.CTkFrame(self.songs_frame)
            f.pack(fill="x", pady=2)
            btn = ctk.CTkButton(f, text="▶️", width=40,
                               command=lambda s=song: self.play(s))
            btn.pack(side="left", padx=5)
            lbl = ctk.CTkLabel(f, text=f"{i}. {song['title']} - {song['artist']}", anchor="w")
            lbl.pack(side="left", fill="x", expand=True, padx=5)
    def play(self, song):
        """Play song"""
        print(f"🎵 Playing: {song['title']}")
        self.music.play_song(song)
    def close(self):
        """Close app"""
        self.active = False
        if self.cap:
            self.cap.release()
        self.root.quit()
        self.root.destroy()
    def run(self):
        """Run app"""
        print("\n🎉 ACCURATE Emotion Detection Ready!")
        print("📷 Click 'Start Camera'")
        print("🎭 Show YOUR expressions - the AI will detect them!")
        print()
        self.root.mainloop()
if __name__ == "__main__":
    try:
        app = AccurateEmotionApp()
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
