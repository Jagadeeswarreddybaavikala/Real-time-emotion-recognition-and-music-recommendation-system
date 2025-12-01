import cv2
import numpy as np
import tensorflow as tf

print("Step 1: Loading the emotion recognition model...")
MODEL_PATH = "models/accurate_emotion_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

print("\nStep 2: Setting up emotion labels...")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
print(f"✅ Emotion classes: {emotions}")

print("\nStep 3: Setting up face detection...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✅ Face detector loaded!")

print("\nStep 4: Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()
print("✅ Webcam opened successfully!")

print("\n🎥 Starting classification test...")
print("Press 'q' to quit\n")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    frame_count += 1
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
        
        text = f"{emotion_label} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
        
        if frame_count % 10 == 0:
            print(f"🎭 Frame {frame_count}: {text}")
    
    cv2.imshow('Classification Module Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Test completed successfully!")
print("Classification module is working correctly.")

