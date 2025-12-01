import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import csv
import os

print("Loading emotion recognition model...")
MODEL_PATH = "models/accurate_emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create results directory
os.makedirs('results', exist_ok=True)

# Create CSV file for logging
timestamp_start = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'results/emotion_log_{timestamp_start}.csv'

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Emotion', 'Confidence (%)', 'Elapsed Time (s)'])

print("🎥 Starting emotion detection...")
print("Press 'q' to quit, 's' to save screenshot\n")

cap = cv2.VideoCapture(0)
start_time = datetime.now()
frame_count = 0
detection_log = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_time = datetime.now()
    elapsed_seconds = (current_time - start_time).total_seconds()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(face_normalized, axis=(0, -1))
        
        prediction = model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion_label = emotions[emotion_idx]
        confidence = prediction[0][emotion_idx] * 100
        
        # Display on frame
        text = f"{emotion_label} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
        
        # Add timestamp
        time_text = f"{elapsed_seconds:.1f}s"
        cv2.putText(frame, time_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 0), 2)
        
        # Log every 10 frames
        if frame_count % 10 == 0:
            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"🎭 [{timestamp_str}] {text} | Time: {elapsed_seconds:.1f}s")
            
            # Save to CSV
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp_str, emotion_label, f"{confidence:.2f}", f"{elapsed_seconds:.2f}"])
            
            detection_log.append({
                'timestamp': timestamp_str,
                'emotion': emotion_label,
                'confidence': confidence,
                'elapsed_time': elapsed_seconds
            })
    
    # Display frame info
    info_text = f"Frame: {frame_count} | Time: {elapsed_seconds:.1f}s"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    
    cv2.imshow('Emotion Detection with Timestamps', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        screenshot_file = f'results/screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        cv2.imwrite(screenshot_file, frame)
        print(f"📸 Screenshot saved: {screenshot_file}")

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Detection completed!")
print(f"📊 Total frames processed: {frame_count}")
print(f"⏱️  Total time: {elapsed_seconds:.2f} seconds")
print(f"📁 Results saved to: {csv_filename}")
print(f"📊 Total detections logged: {len(detection_log)}")

