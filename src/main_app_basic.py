"""
Real-time Emotion Recognition Music Recommendation System
BASIC VERSION - No TensorFlow Required
Works with face detection + emotion simulation
"""

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import cv2
import numpy as np
import threading
import queue
import time
from PIL import Image, ImageTk
import pygame
import json
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from music_system import MusicRecommendationSystem
    print("✅ Music system imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class BasicEmotionModel:
    """Simple emotion model without TensorFlow"""
    def __init__(self):
        self.emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad',
            5: 'Surprise', 6: 'Neutral', 7: 'Contempt', 8: 'Excited', 9: 'Confused'
        }
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = True
        self.last_emotion = 'Neutral'
        self.last_change_time = time.time()
        self.emotion_cycle = ['Neutral', 'Happy', 'Surprise', 'Excited', 'Sad', 'Angry', 'Fear', 'Confused']
        self.cycle_index = 0
        
    def create_model(self):
        """Create basic model"""
        print("✅ Basic emotion model created")
        self.model = True
        return True
    
    def detect_faces(self, frame):
        """Detect faces using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces, gray
    
    def predict_emotion(self, face_image):
        """Predict emotion with cycling"""
        current_time = time.time()
        
        
        if current_time - self.last_change_time > 4:
            self.cycle_index = (self.cycle_index + 1) % len(self.emotion_cycle)
            self.last_emotion = self.emotion_cycle[self.cycle_index]
            self.last_change_time = current_time
            print(f"🎭 Emotion changed to: {self.last_emotion}")
        
        emotion_idx = list(self.emotion_labels.values()).index(self.last_emotion)
        confidence = 0.85  
        
        return self.last_emotion, confidence, emotion_idx
    
    def load_model(self, model_path):
        """Load basic model"""
        self.model = True
        return True

class EmotionMusicApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🎵 Emotion Music Recommendation System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
       
        self.emotion_model = BasicEmotionModel()
        self.music_system = MusicRecommendationSystem()
        
       
        self.cap = None
        self.camera_active = False
        self.current_emotion = "Neutral"
        self.emotion_confidence = 0.0
        self.emotion_history = []
        self.frame_queue = queue.Queue(maxsize=5)
        
       
        self.video_label = None
        self.current_recommendations = []
        self.is_playing = False
        self.current_volume = 0.7
        
      
        self.detection_thread = None
        self.camera_thread = None
        self.stop_threads = False
        
        
        self.setup_ui()
        self.emotion_model.create_model()
        
       
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        """Setup the modern UI"""
        
        
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        
        self.left_frame = ctk.CTkFrame(self.root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)
        
        
        self.camera_control_frame = ctk.CTkFrame(self.left_frame)
        self.camera_control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.camera_status_label = ctk.CTkLabel(
            self.camera_control_frame, 
            text="📷 Camera: OFF", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.camera_status_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.start_camera_btn = ctk.CTkButton(
            self.camera_control_frame,
            text="▶️ Start Camera",
            command=self.toggle_camera,
            font=ctk.CTkFont(size=14),
            height=40
        )
        self.start_camera_btn.grid(row=0, column=1, padx=10, pady=5)
        
        
        self.video_frame = ctk.CTkFrame(self.left_frame)
        self.video_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="🎭 Camera feed will appear here\n\n👆 Click 'Start Camera' to begin\n\n✨ Watch real-time emotion detection!",
            font=ctk.CTkFont(size=16),
            width=500,
            height=400
        )
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        
        self.emotion_frame = ctk.CTkFrame(self.left_frame)
        self.emotion_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        self.emotion_title = ctk.CTkLabel(
            self.emotion_frame,
            text="🎭 Current Emotion",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.emotion_title.grid(row=0, column=0, columnspan=2, padx=10, pady=5)
        
        self.emotion_label = ctk.CTkLabel(
            self.emotion_frame,
            text="😐 Neutral",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#00ff00"
        )
        self.emotion_label.grid(row=1, column=0, padx=10, pady=5)
        
        self.confidence_label = ctk.CTkLabel(
            self.emotion_frame,
            text="Confidence: 0%",
            font=ctk.CTkFont(size=14)
        )
        self.confidence_label.grid(row=1, column=1, padx=10, pady=5)
        
        
        self.right_frame = ctk.CTkFrame(self.root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)
        
       
        self.music_control_frame = ctk.CTkFrame(self.right_frame)
        self.music_control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.music_title = ctk.CTkLabel(
            self.music_control_frame,
            text="🎵 Music Recommendations",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.music_title.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        
        
        self.play_btn = ctk.CTkButton(
            self.music_control_frame,
            text="▶️",
            command=self.toggle_playback,
            font=ctk.CTkFont(size=16),
            width=50,
            height=40
        )
        self.play_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.stop_btn = ctk.CTkButton(
            self.music_control_frame,
            text="⏹️",
            command=self.stop_music,
            font=ctk.CTkFont(size=16),
            width=50,
            height=40
        )
        self.stop_btn.grid(row=1, column=1, padx=5, pady=5)
        
        
        self.volume_label = ctk.CTkLabel(
            self.music_control_frame,
            text="🔊 Volume:",
            font=ctk.CTkFont(size=14)
        )
        self.volume_label.grid(row=1, column=2, padx=5, pady=5)
        
        self.volume_slider = ctk.CTkSlider(
            self.music_control_frame,
            from_=0,
            to=1,
            command=self.set_volume,
            width=150
        )
        self.volume_slider.grid(row=1, column=3, padx=5, pady=5)
        self.volume_slider.set(self.current_volume)
        
      
        self.recommendations_frame = ctk.CTkScrollableFrame(self.right_frame)
        self.recommendations_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.recommendations_frame.grid_columnconfigure(0, weight=1)
        
        
        self.update_recommendations()
        
        print("✅ GUI setup completed!")
        
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera and detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera")
                return
            
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_active = True
            self.stop_threads = False
            
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            
            self.camera_thread.start()
            self.detection_thread.start()
            
            
            self.start_camera_btn.configure(text="⏹️ Stop Camera")
            self.camera_status_label.configure(text="📷 Camera: ON", text_color="#00ff00")
            
            print("✅ Camera started successfully!")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera"""
        self.camera_active = False
        self.stop_threads = True
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_camera_btn.configure(text="▶️ Start Camera")
        self.camera_status_label.configure(text="📷 Camera: OFF", text_color="#ff0000")
        self.video_label.configure(image=None, text="🎭 Camera stopped\n\n👆 Click 'Start Camera' to restart")
        
        print("✅ Camera stopped")
    
    def camera_loop(self):
        """Camera capture loop"""
        while self.camera_active and not self.stop_threads:
            try:
                ret, frame = self.cap.read()
                if ret:
                    
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                    
                    
                    self.update_video_display(frame)
                
                time.sleep(1/30)  
                
            except Exception as e:
                print(f"❌ Camera loop error: {e}")
                break
    
    def detection_loop(self):
        """Emotion detection loop"""
        while self.camera_active and not self.stop_threads:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    
                    faces, gray = self.emotion_model.detect_faces(frame)
                    
                    if len(faces) > 0:
                        
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = largest_face
                        
                        
                        face_roi = gray[y:y+h, x:x+w]
                        emotion, confidence, _ = self.emotion_model.predict_emotion(face_roi)
                        
                        
                        self.update_emotion(emotion, confidence)
                
                time.sleep(0.1)  
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
                time.sleep(0.5)
    
    def update_video_display(self, frame):
        """Update video display"""
        try:
            
            faces, gray = self.emotion_model.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{self.current_emotion}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            
            display_frame = cv2.resize(frame, (500, 400))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            
            self.root.after(0, lambda: self.video_label.configure(image=photo, text=""))
            self.root.after(0, lambda: setattr(self.video_label, 'image', photo))
            
        except Exception as e:
            print(f"❌ Video display error: {e}")
    
    def update_emotion(self, emotion, confidence):
        """Update current emotion and UI"""
        self.current_emotion = emotion
        self.emotion_confidence = confidence
        
        
        self.emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        
        if len(self.emotion_history) > 10:
            self.emotion_history.pop(0)
        
        
        self.root.after(0, self.update_emotion_display)
        self.root.after(0, self.update_recommendations)
    
    def update_emotion_display(self):
        """Update emotion display"""
        
        emotion_emojis = {
            'Happy': '😊', 'Sad': '😢', 'Angry': '😠', 'Fear': '😨',
            'Surprise': '😲', 'Disgust': '🤢', 'Neutral': '😐',
            'Contempt': '😤', 'Excited': '🤩', 'Confused': '😕'
        }
        
        
        emotion_colors = {
            'Happy': '#00ff00', 'Sad': '#0080ff', 'Angry': '#ff4444',
            'Fear': '#ff8800', 'Surprise': '#ffff00', 'Disgust': '#8800ff',
            'Neutral': '#888888', 'Contempt': '#ff0088', 'Excited': '#00ffff',
            'Confused': '#ff8888'
        }
        
        emoji = emotion_emojis.get(self.current_emotion, '😐')
        color = emotion_colors.get(self.current_emotion, '#ffffff')
        
        self.emotion_label.configure(
            text=f"{emoji} {self.current_emotion}", 
            text_color=color
        )
        self.confidence_label.configure(
            text=f"Confidence: {self.emotion_confidence:.0%}"
        )
    
    def update_recommendations(self):
        """Update music recommendations"""
        songs = self.music_system.get_all_songs_for_emotion(self.current_emotion)
        
        
        for widget in self.recommendations_frame.winfo_children():
            widget.destroy()
        
        
        title_label = ctk.CTkLabel(
            self.recommendations_frame,
            text=f"🎵 Songs for {self.current_emotion} ({len(songs)} songs)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=10)
        
        
        for i, song in enumerate(songs):
            song_frame = ctk.CTkFrame(self.recommendations_frame)
            song_frame.pack(fill="x", padx=5, pady=2)
            
            
            play_button = ctk.CTkButton(
                song_frame,
                text="▶️",
                command=lambda s=song: self.play_song(s),
                width=40,
                height=30
            )
            play_button.pack(side="left", padx=5, pady=5)
            
            
            song_text = f"{song['title']}\n{song['artist']} • {song['genre']}"
            song_label = ctk.CTkLabel(
                song_frame,
                text=song_text,
                font=ctk.CTkFont(size=11),
                justify="left"
            )
            song_label.pack(side="left", padx=5, pady=5, fill="x", expand=True)
    
    def play_song(self, song_info):
        """Play selected song"""
        try:
            self.music_system.play_song(song_info)
            self.play_btn.configure(text="⏸️")
            self.is_playing = True
            print(f"🎵 Playing: {song_info['title']} by {song_info['artist']}")
        except Exception as e:
            print(f"❌ Playback error: {e}")
    
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.music_system.stop_song()
            self.play_btn.configure(text="▶️")
            self.is_playing = False
        else:
            print("Select a song to play")
    
    def stop_music(self):
        """Stop music"""
        self.music_system.stop_song()
        self.play_btn.configure(text="▶️")
        self.is_playing = False
    
    def set_volume(self, value):
        """Set volume"""
        self.current_volume = float(value)
        self.music_system.set_volume(self.current_volume)
    
    def on_closing(self):
        """Handle app closing"""
        self.stop_camera()
        if self.music_system:
            self.music_system.stop_song()
        try:
            pygame.mixer.quit()
        except:
            pass
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the app"""
        try:
            print("🚀 Emotion Music System started!")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()

def main():
    """Main function"""
    try:
        app = EmotionMusicApp()
        app.run()
    except Exception as e:
        print(f"❌ Application error: {e}")

if __name__ == "__main__":
    main()
