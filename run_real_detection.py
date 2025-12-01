
"""
Real Facial Expression Detection - NO TensorFlow Required!
Uses OpenCV smile/eye detection to respond to YOUR expressions
"""

import sys
import os
from pathlib import Path

def main():
    print("🎭 Real Facial Expression Detection")
    print("=" * 50)
    print("😊 Smile → Happy")
    print("😐 Neutral face → Neutral")
    print("👀 Wide eyes → Surprise")
    print("😢 Sad face → Sad")
    print()
    
  
    sys.path.insert(0, 'src')
    
    try:
        from emotion_model_smart import SmartEmotionDetector
        from music_system import MusicRecommendationSystem
        import customtkinter as ctk
        import cv2
        import threading
        import queue
        import time
        from PIL import Image, ImageTk
        
        print("✅ All imports successful!")
        print("🚀 Starting app...\n")
      
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        class RealEmotionApp:
            def __init__(self):
                self.root = ctk.CTk()
                self.root.title("🎭 Real Emotion Detection + Music")
                self.root.geometry("1400x900")
                
               
                self.detector = SmartEmotionDetector()
                self.detector.create_model()
                self.music = MusicRecommendationSystem()
                
             
                self.cap = None
                self.camera_active = False
                self.current_emotion = "Neutral"
                self.frame_queue = queue.Queue(maxsize=2)
                
                self.setup_ui()
                self.root.protocol("WM_DELETE_WINDOW", self.on_close)
                
            def setup_ui(self):
                
                left = ctk.CTkFrame(self.root)
                left.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
                self.root.grid_columnconfigure(0, weight=1)
                self.root.grid_columnconfigure(1, weight=1)
                self.root.grid_rowconfigure(0, weight=1)
                
               
                self.cam_btn = ctk.CTkButton(
                    left, text="📷 Start Camera",
                    command=self.toggle_camera,
                    font=("Arial", 16), height=50
                )
                self.cam_btn.pack(pady=20)
                
                
                self.video_label = ctk.CTkLabel(left, text="Camera will appear here")
                self.video_label.pack(expand=True, fill="both", padx=10, pady=10)
                
                
                self.emotion_label = ctk.CTkLabel(
                    left, text="😐 Neutral",
                    font=("Arial", 32, "bold")
                )
                self.emotion_label.pack(pady=20)
                
                
                right = ctk.CTkFrame(self.root)
                right.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
                
                title = ctk.CTkLabel(right, text="🎵 Music Recommendations", 
                                    font=("Arial", 20, "bold"))
                title.pack(pady=20)
                
                
                self.songs_frame = ctk.CTkScrollableFrame(right)
                self.songs_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                self.update_songs()
                
            def toggle_camera(self):
                if not self.camera_active:
                    self.cap = cv2.VideoCapture(0)
                    if self.cap.isOpened():
                        self.camera_active = True
                        self.cam_btn.configure(text="⏹️ Stop Camera")
                        threading.Thread(target=self.camera_loop, daemon=True).start()
                        threading.Thread(target=self.detect_loop, daemon=True).start()
                        print("✅ Camera started - SMILE to test!")
                else:
                    self.camera_active = False
                    if self.cap:
                        self.cap.release()
                    self.cam_btn.configure(text="📷 Start Camera")
            
            def camera_loop(self):
                while self.camera_active:
                    ret, frame = self.cap.read()
                    if ret:
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame.copy())
                        self.show_frame(frame)
                    time.sleep(1/30)
            
            def detect_loop(self):
                while self.camera_active:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get()
                        faces, gray = self.detector.detect_faces(frame)
                        
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_roi = gray[y:y+h, x:x+w]
                            emotion, conf, _ = self.detector.predict_emotion(face_roi)
                            
                            if emotion != self.current_emotion:
                                print(f"🎭 Detected: {emotion} ({conf:.0%})")
                                self.current_emotion = emotion
                                self.root.after(0, self.update_emotion)
                    time.sleep(0.1)
            
            def show_frame(self, frame):
                faces, _ = self.detector.detect_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, self.current_emotion, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                frame = cv2.resize(frame, (640, 480))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(img)
                
                self.root.after(0, lambda: self.video_label.configure(image=photo, text=""))
                self.root.after(0, lambda: setattr(self.video_label, 'image', photo))
            
            def update_emotion(self):
                emojis = {
                    'Happy': '😊', 'Sad': '😢', 'Angry': '😠',
                    'Surprise': '😲', 'Neutral': '😐', 'Fear': '😨',
                    'Excited': '🤩', 'Disgust': '🤢'
                }
                emoji = emojis.get(self.current_emotion, '😐')
                self.emotion_label.configure(text=f"{emoji} {self.current_emotion}")
                self.update_songs()
            
            def update_songs(self):
                for widget in self.songs_frame.winfo_children():
                    widget.destroy()
                
                songs = self.music.get_all_songs_for_emotion(self.current_emotion)
                
                for song in songs[:10]:
                    frame = ctk.CTkFrame(self.songs_frame)
                    frame.pack(fill="x", pady=2)
                    
                    btn = ctk.CTkButton(frame, text="▶️", width=40,
                                       command=lambda s=song: self.play(s))
                    btn.pack(side="left", padx=5)
                    
                    label = ctk.CTkLabel(frame, 
                                        text=f"{song['title']} - {song['artist']}",
                                        anchor="w")
                    label.pack(side="left", fill="x", expand=True, padx=5)
            
            def play(self, song):
                print(f"🎵 Would play: {song['title']}")
                self.music.play_song(song)
            
            def on_close(self):
                self.camera_active = False
                if self.cap:
                    self.cap.release()
                self.root.quit()
                self.root.destroy()
            
            def run(self):
                print("🎉 App is running!")
                print("👉 Click 'Start Camera' button")
                print("😊 Then SMILE BIG to test detection!\n")
                self.root.mainloop()
        
        
        app = RealEmotionApp()
        app.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMissing package. Install with:")
        print("python3.11 -m pip install customtkinter pillow pygame opencv-python")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
