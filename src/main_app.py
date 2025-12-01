"""
Real-time Emotion Recognition Music Recommendation System
Modern GUI with smooth animations and real-time emotion detection
Optimized for Mac M2 with lag-free performance
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
    from emotion_model import EmotionRecognitionModel
    from music_system import MusicRecommendationSystem
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure emotion_model.py and music_system.py are in the src directory")

ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")  

class EmotionMusicApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Emotion-Based Music Recommendation System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
      
        self.emotion_model = EmotionRecognitionModel()
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
        self.load_or_create_model()
        
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        """Setup the modern UI with smooth animations"""
        
        
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
            text="Camera: OFF", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.camera_status_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.start_camera_btn = ctk.CTkButton(
            self.camera_control_frame,
            text="Start Camera",
            command=self.toggle_camera,
            font=ctk.CTkFont(size=14),
            height=40
        )
        self.start_camera_btn.grid(row=0, column=1, padx=10, pady=5)
        
        
        self.video_frame = ctk.CTkFrame(self.left_frame)
        self.video_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="Camera feed will appear here\nClick 'Start Camera' to begin",
            font=ctk.CTkFont(size=14),
            width=500,
            height=400
        )
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        print("✅ GUI setup completed successfully!")
        
    def load_or_create_model(self):
        """Load existing model or create new one"""
        model_path = "models/emotion_model.h5"
        
        if os.path.exists(model_path):
            if self.emotion_model.load_model(model_path):
                print("✅ Model loaded successfully!")
                return
        
        
        try:
            self.emotion_model.create_model()
            print("✅ New model created. Training recommended.")
        except Exception as e:
            print(f"❌ Error creating model: {e}")
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera and detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Could not open camera")
                return
            
            self.camera_active = True
            print("✅ Camera started successfully!")
            
        except Exception as e:
            print(f"❌ Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop the camera and detection"""
        self.camera_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("✅ Camera stopped")
    
    def on_closing(self):
        """Handle application closing"""
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
        """Start the application"""
        try:
            print("🚀 Application started")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()

def main():
    """Main function to run the application"""
    try:
       
        app = EmotionMusicApp()
        app.run()
    except Exception as e:
        print(f"❌ Application error: {e}")

if __name__ == "__main__":
    main()
