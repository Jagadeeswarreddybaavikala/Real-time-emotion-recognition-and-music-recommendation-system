![banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=160&section=header&text=Emotion%20Recognition%20+%20Music%20AI&fontSize=34&animation=twinkling)




🎵 Real-Time Emotion Recognition & Music Recommendation System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A machine learning–powered system that detects human emotions in real time using facial expressions and automatically recommends music based on the detected emotion.
This project combines computer vision, deep learning, and intelligent music mapping to create an interactive, mood-aware experience.

🚀 Features

✔️ Real-time face detection using webcam
✔️ Deep learning model trained on FER-2013 dataset
✔️ Emotion classification (Happy, Sad, Angry, Neutral, etc.)
✔️ Music recommendation based on emotional state
✔️ Visualization of training results (accuracy, loss graphs)
✔️ Modular and scalable codebase
✔️ Fully reproducible environment using requirements.txt

🧠 Supported Emotions
Emotion	Possible Music Type
😊 Happy	Pop / Energetic / EDM
😢 Sad	Calm / LoFi / Relaxation
😡 Angry	Rock / Metal / Workout
😐 Neutral	Ambient / Soft Instrumental
😲 Surprise	Trending playlist (optional)

You can modify mappings in spotify_music.py.

🧱 Tech Stack
Category	Tools
Language	Python 3
ML Framework	TensorFlow / Keras
Computer Vision	OpenCV
Data Handling	NumPy, Pandas
Visualization	Matplotlib / Seaborn
Deployment Mode	Local Script (Future: Streamlit / Flask UI)
📁 Folder Structure
Real-time-emotion-recognition-and-music-recommendation-system/
├── src/                   
├── data/                      
├── models/                   
├── project_results/        
├── enhanced_emotion_detection.py
├── train_fer2013_model.py
├── run_real_detection.py
├── run_trained_app.py
├── spotify_music.py
├── plot_training_history.py
├── generate_report.py
├── test_classification.py
├── requirements.txt
└── README.md

🛠 Installation
1️⃣ Clone the repository
git clone https://github.com/Jagadeeswarreddybaavikala/Real-time-emotion-recognition-and-music-recommendation-system.git
cd Real-time-emotion-recognition-and-music-recommendation-system

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🧪 Train the Model
python train_fer2013_model.py


This script will:

Train the CNN on FER-2013

Save trained model in models/

Generate logs in project_results/

Optional: visualize training:

python plot_training_history.py

🎥 Run Real-Time Detection
python run_real_detection.py


This opens the webcam, detects the face, and displays live emotion predictions.

🎧 Run Full Emotion→Music System
python run_trained_app.py


This will:

Detect your face

Predict your emotion

Recommend music using logic in spotify_music.py

(Music source can be Spotify API, YouTube links, local MP3s, etc.) 

🔮 Future Enhancements

🔹 Deploy as a web app using Flask / Streamlit
🔹 Add voice emotion + sentiment analysis
🔹 Use transfer learning (ResNet / EfficientNet)
🔹 Enable Spotify OAuth live control
🔹 Add multi-user emotion awareness

👤 Author

BAAVIKALA JAGADEESWAR REDDY
🎓 SDE | Developer | Innovator
🔗 GitHub: Jagadeeswarreddybaavikala

📜 License

This project is open-source and available under the MIT License.

MIT License — feel free to use, modify, and improve.

⭐ If you like this project, please star the repository 💙
