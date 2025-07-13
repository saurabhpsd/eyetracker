import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from deepface import DeepFace

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize OpenCV for webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Data storage for features
data = []
gaze_points = deque(maxlen=100)  # For heatmap
emotion_counts = {'angry': 0, 'fear': 0, 'neutral': 0, 'sad': 0, 'disgust': 0, 'happy': 0, 'surprise': 0}
blink_count = 0
last_blink_time = time.time()
fixation_start = time.time()
fixation_duration = 0
last_gaze = None
saccade_count = 0
frame_count = 0
start_time = time.time()

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.23
SACCADE_THRESHOLD = 0.05

def calculate_eye_aspect_ratio(eye_points, landmarks, image_shape):
    """Calculate the eye aspect ratio (EAR) to detect blinks."""
    h, w = image_shape[:2]
    p1 = np.array([landmarks[eye_points[0]].x * w, landmarks[eye_points[0]].y * h])
    p2 = np.array([landmarks[eye_points[1]].x * w, landmarks[eye_points[1]].y * h])
    p3 = np.array([landmarks[eye_points[2]].x * w, landmarks[eye_points[2]].y * h])
    p4 = np.array([landmarks[eye_points[3]].x * w, landmarks[eye_points[3]].y * h])
    p5 = np.array([landmarks[eye_points[4]].x * w, landmarks[eye_points[4]].y * h])
    p6 = np.array([landmarks[eye_points[5]].x * w, landmarks[eye_points[5]].y * h])
    
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    hor = np.linalg.norm(p1 - p4)
    ear = (vert1 + vert2) / (2.0 * hor) if hor != 0 else 0
    return ear

def get_gaze_direction(landmarks, image_shape):
    """Estimate gaze direction based on pupil position."""
    h, w = image_shape[:2]
    left_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
    right_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])
    gaze_center = (left_pupil + right_pupil) / 2
    return gaze_center

def get_emotion_probabilities(frame):
    """Detect emotions using DeepFace and return probabilities."""
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
        return result[0]['emotion']
    except:
        return {'angry': 0, 'fear': 0, 'neutral': 0, 'sad': 0, 'disgust': 0, 'happy': 0, 'surprise': 0}

# Main loop for real-time tracking
while cap.isOpened() and frame_count < 300:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    frame_count += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    # Emotion detection
    emotion_probs = get_emotion_probabilities(frame)
    dominant_emotion = max(emotion_probs, key=emotion_probs.get)
    emotion_counts[dominant_emotion] += 1
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate Eye Aspect Ratio for blink detection
            left_ear = calculate_eye_aspect_ratio(LEFT_EYE, face_landmarks.landmark, frame.shape)
            right_ear = calculate_eye_aspect_ratio(RIGHT_EYE, face_landmarks.landmark, frame.shape)
            ear = (left_ear + right_ear) / 2.0
            
            # Detect blink
            if ear < BLINK_THRESHOLD:
                if time.time() - last_blink_time > 0.5:
                    blink_count += 1
                    last_blink_time = time.time()
            
            # Calculate gaze direction
            gaze = get_gaze_direction(face_landmarks.landmark, frame.shape)
            gaze_points.append(gaze)
            
            # Detect saccades
            if last_gaze is not None and np.linalg.norm(gaze - last_gaze) > SACCADE_THRESHOLD * frame.shape[1]:
                saccade_count += 1
                fixation_duration = time.time() - fixation_start
                fixation_start = time.time()
            else:
                fixation_duration = time.time() - fixation_start
            last_gaze = gaze
            
            # Draw landmarks and gaze point
            for idx in LEFT_EYE + RIGHT_EYE + [468, 473]:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(gaze[0]), int(gaze[1])), 5, (0, 0, 255), -1)
            
            # Store features
            label = 1 if time.time() - start_time < 5 else 0
            features = {
                'fixation_duration': fixation_duration,
                'saccade_frequency': saccade_count / (frame_count / 30.0) if frame_count > 0 else 0,
                'blink_rate': blink_count / ((time.time() - start_time) / 60.0) if time.time() - start_time > 0 else 0,
                'gaze_x': gaze[0] / frame.shape[1],
                'gaze_y': gaze[1] / frame.shape[0],
                'emotion_angry': emotion_probs['angry'],
                'emotion_fear': emotion_probs['fear'],
                'emotion_neutral': emotion_probs['neutral'],
                'emotion_sad': emotion_probs['sad'],
                'emotion_disgust': emotion_probs['disgust'],
                'emotion_happy': emotion_probs['happy'],
                'emotion_surprise': emotion_probs['surprise'],
                'label': label
            }
            data.append(features)
    
    # Display frame
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Saccades: {saccade_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Eye Tracker with Emotion', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

# Save data to CSV
if data:
    df = pd.DataFrame(data)
    df.to_csv('eye_tracking_emotion_data.csv', index=False)
else:
    print("No data collected. Check webcam or face detection.")


# Train Random Forest Classifier
X = df[['fixation_duration', 'saccade_frequency', 'blink_rate', 'gaze_x', 'gaze_y',
        'emotion_angry', 'emotion_fear', 'emotion_neutral', 'emotion_sad',
        'emotion_disgust', 'emotion_happy', 'emotion_surprise']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Distracted', 'Focused']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Generate gaze heatmap
plt.figure(figsize=(10, 6))
gaze_x = [p[0] for p in gaze_points]
gaze_y = [p[1] for p in gaze_points]
if gaze_x and gaze_y:
    sns.kdeplot(x=gaze_x, y=gaze_y, cmap="Reds", fill=True)
    plt.title('Gaze Heatmap')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('gaze_heatmap.png')
    plt.close()
else:
    print("No gaze points collected for heatmap.")

# Generate attention score plot
attention_scores = clf.predict_proba(X)[:, 1]
plt.figure(figsize=(10, 6))
plt.plot(attention_scores, label='Attention Score', color='#1f77b4')
plt.title('Attention Score Over Time (with Emotion Fusion)')
plt.xlabel('Sample')
plt.ylabel('Attention Probability')
plt.legend()
plt.savefig('attention_score.png')
plt.close()

# Generate emotion distribution plot
plt.figure(figsize=(10, 6))
sns.barplot(x=list(emotion_counts.values()), y=list(emotion_counts.keys()), palette='viridis')
plt.title('Emotion Distribution')
plt.xlabel('Count')
plt.ylabel('Emotion')
plt.savefig('emotion_distribution.png')
plt.close()