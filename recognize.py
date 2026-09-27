import os
import urllib.request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

MODEL_PATH = os.path.join(BASE_DIR, "final_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")

print("=" * 60)
print(" REAL-TIME FACE RECOGNITION ")
print("=" * 60)

# Check Haar cascade
classifier = cv2.CascadeClassifier(CASCADE_PATH)
if classifier.empty():
    print(f"Error: Unable to load Haar cascade from '{CASCADE_PATH}'")
    exit(1)

# Check trained model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Run 'train.py' first.")
    exit(1)

print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Load dynamic classes
if os.path.exists(CLASSES_PATH):
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    print(f"Loaded classes: {list(classes)}")
else:
    print("Warning: classes.npy not found. Falling back to default labels.")
    classes = np.array([f"Person_{i}" for i in range(model.output_shape[-1])])

CONFIDENCE_THRESHOLD = 50.0  # Percentage minimum to accept recognition

def preprocess(img):
    """Consistent preprocessing: Grayscale -> Resize (100, 100) -> Hist Eq -> Normalize"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Shape: (1, 100, 100, 1)
    return img

print("\nSelect Camera Source:")
print(" [0] Built-in / USB Webcam (Default)")
print(" [1] IP Camera URL (e.g. http://192.168.1.X:8080/shot.jpg)")
source_choice = input("Enter choice (0 or 1, default 0): ").strip()

camera_url = None
cap = None

if source_choice == '1':
    camera_url = input("Enter IP Camera shot URL (e.g. http://192.168.1.X:8080/shot.jpg): ").strip()
    print(f"Connecting to IP Camera stream: {camera_url}...")
else:
    print("Opening webcam (device index 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open device 0. Trying device 1...")
        cap = cv2.VideoCapture(1)

print("\nRecognition running. Press 'q' to quit.")

def get_frame():
    if camera_url:
        try:
            req = urllib.request.urlopen(camera_url, timeout=5)
            img_arr = np.array(bytearray(req.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)
            return frame
        except Exception as e:
            print(f"Camera stream error: {e}")
            return None
    elif cap and cap.isOpened():
        ret, frame = cap.read()
        return frame if ret else None
    return None

while True:
    frame = get_frame()
    if frame is None:
        cv2.waitKey(200)
        continue

    h_frame, w_frame = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        y1, y2 = max(0, y), min(h_frame, y + h)
        x1, x2 = max(0, x), min(w_frame, x + w)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Predict with trained CNN model
        processed = preprocess(face_crop)
        preds = model.predict(processed, verbose=0)[0]
        best_idx = int(np.argmax(preds))
        confidence = float(preds[best_idx]) * 100.0

        if best_idx < len(classes) and confidence >= CONFIDENCE_THRESHOLD and classes[best_idx] != "Unknown":
            label_text = f"{classes[best_idx]}: {confidence:.1f}%"
            box_color = (0, 255, 0)      # Green for recognized
        else:
            label_text = f"Unknown ({confidence:.1f}%)"
            box_color = (0, 0, 255)      # Red for unknown

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Draw background label plate
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(y1 - 10, text_h + 10)
        cv2.rectangle(frame, (x1, label_y - text_h - 4), (x1 + text_w + 4, label_y + baseline), box_color, -1)
        cv2.putText(frame, label_text, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, "Press 'q' to quit", (20, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
print("Recognition stopped.")
