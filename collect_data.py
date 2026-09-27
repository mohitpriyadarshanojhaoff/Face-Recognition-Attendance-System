import os
import urllib.request
import cv2
import numpy as np

# Resolve base paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

OUTPUT_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

classifier = cv2.CascadeClassifier(CASCADE_PATH)
if classifier.empty():
    print(f"Error: Could not load Haar cascade classifier from {CASCADE_PATH}")
    exit(1)

print("=" * 50)
print(" FACE DATA COLLECTION ")
print("=" * 50)

# Ask for the person's name upfront
person_name = input("Enter the person's name (e.g. Mohit, Alex): ").strip()
while not person_name:
    person_name = input("Name cannot be empty. Please enter a name: ").strip()

print("\nSelect Camera Source:")
print(" [0] Built-in / USB Webcam (Default)")
print(" [1] IP Camera URL (e.g., http://192.168.1.X:8080/shot.jpg)")
source_choice = input("Enter choice (0 or 1, default 0): ").strip()

camera_url = None
cap = None

if source_choice == '1':
    camera_url = input("Enter IP Camera shot URL (e.g. http://192.168.1.X:8080/shot.jpg): ").strip()
    print(f"Connecting to IP Camera: {camera_url}...")
else:
    print("Opening default webcam (device 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open webcam at index 0. Trying index 1...")
        cap = cv2.VideoCapture(1)

TARGET_SAMPLES = 100
captured_faces = []
sample_count = 0

print(f"\nCollecting {TARGET_SAMPLES} face samples for '{person_name}'...")
print("Look directly at the camera. Press 'q' anytime to quit.\n")

def read_frame():
    if camera_url:
        try:
            req = urllib.request.urlopen(camera_url, timeout=5)
            img_arr = np.array(bytearray(req.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)
            return frame
        except Exception as e:
            print(f"Error reading from IP camera: {e}")
            return None
    elif cap and cap.isOpened():
        ret, frame = cap.read()
        return frame if ret else None
    return None

while len(captured_faces) < TARGET_SAMPLES:
    frame = read_frame()
    if frame is None:
        print("Waiting for video stream frame...")
        cv2.waitKey(200)
        continue

    h_frame, w_frame = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_points = classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(face_points) > 0:
        # Pick the largest face in frame
        face_points = sorted(face_points, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = face_points[0]

        # Safe bounding box slicing
        y1, y2 = max(0, y), min(h_frame, y + h)
        x1, x2 = max(0, x), min(w_frame, x + w)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size > 0:
            captured_faces.append(face_crop)
            sample_count += 1
            print(f"Captured: {sample_count}/{TARGET_SAMPLES}")
            cv2.imshow("Face Crop Preview", face_crop)

            # Draw green bounding box on main frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Progress text overlay
    progress_text = f"Sample: {len(captured_faces)}/{TARGET_SAMPLES} - {person_name}"
    cv2.putText(frame, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("\nCollection cancelled by user.")
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()

if len(captured_faces) == TARGET_SAMPLES:
    print(f"\nSaving {TARGET_SAMPLES} samples to '{OUTPUT_DIR}'...")
    for i, face_img in enumerate(captured_faces):
        filename = f"{person_name}_{i}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, face_img)
    print(f"Successfully saved {TARGET_SAMPLES} samples for '{person_name}'.")
    print("Next step: Run 'consolidated_data.py' to process the new dataset.")
else:
    print(f"\nCaptured only {len(captured_faces)}/{TARGET_SAMPLES} samples. Discarding to maintain consistency.")
