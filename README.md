# Eye Movement Tracker with Emotion Fusion for Cognitive State Analysis

This project implements a real-time system that fuses **eye movement tracking** with **facial emotion detection** to analyze a user’s **cognitive state** (e.g., focused or distracted). Using **MediaPipe**, **OpenCV**, and **DeepFace**, features like blink rate, saccades, gaze coordinates, and emotions are extracted and classified with a **Random Forest classifier**.

## 📌 Features

- Real-time eye tracking using MediaPipe Face Mesh
- Blink detection via Eye Aspect Ratio (EAR)
- Gaze tracking and saccade detection
- Emotion recognition using DeepFace
- Fusion of eye movement and emotion features
- Cognitive state classification (Focused / Distracted)
- Gaze heatmap and attention score visualizations
- Accuracy: **97%** with high precision and perfect recall for focused states

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** OpenCV, MediaPipe, DeepFace, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
- **IDE:** Jupyter Notebook / VS Code

## 📊 Methodology

1. **Video Capture:** Captures live feed using OpenCV.
2. **Facial Landmark Detection:** Uses MediaPipe Face Mesh to detect key points.
3. **Feature Extraction:** Computes:
   - EAR for blink detection
   - Gaze and pupil coordinates
   - Fixation duration and saccade frequency
4. **Emotion Detection:** Uses DeepFace to classify basic emotions.
5. **Labeling:** Simulates focus/distracted labels for training.
6. **Model Training:** Trains a Random Forest classifier.
7. **Visualization:** Heatmaps and attention trends via Matplotlib/Seaborn.

## 🚀 How to Run
# Clone the repository
git clone https://github.com/your-username/eye-tracker-cognitive-analysis.git
cd eye-tracker-cognitive-analysis

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
⚠️ Ensure your webcam is enabled and working before running the code.

📈 Results
Metric	Value
Accuracy	97%
Precision	96.43%
Recall (Focused)	100%

Confusion Matrix:

lua
Copy
Edit
[[32, 1],   # Distracted (TN, FP)
 [ 0, 27]]  # Focused    (FN, TP)

💡 Applications
🎓 E-learning: Monitor student attention in real time

🚗 Driver Monitoring: Detect drowsiness or distraction

🧠 Healthcare: Assist in diagnosis of neurological conditions

🎮 Gaming: Enable gaze-based controls

📊 UX Testing: Measure user attention to improve interfaces

📌 Limitations
Single face tracking only

Lighting and head pose sensitivity

Uses simulated labels

Short duration data collection

🧠 Future Enhancements
Real-world label collection

Real-time feedback in GUI

Multi-face tracking support

Extended dataset for generalization

📚 References
Includes research papers from ETRA 2024–2025, Behavior Research Methods, Applied Sciences, and more. See the [report](./ai final report.pdf) for full citations.

👨‍💻 Contributors:
Saurabh Prasad
Rohan
Shruti Sinha
