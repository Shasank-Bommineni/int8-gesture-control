import os
import cv2
import time
import glob
import numpy as np
import pandas as pd
import mediapipe as mp

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLES_PER_GESTURE = 400
COUNTDOWN_SECONDS = 3
FRAME_STRIDE = 2  # Collect 1 frame every N frames (configurable sampling rate)
DUPLICATE_THRESHOLD = 0.005  # Normalized distance threshold for duplicate rejection

GESTURES = {
    0: "Open Palm (Rotate)",
    1: "Fist (Grab/Move)",
    2: "Pinch (Zoom)",
    3: "Point (Select)",
    4: "Flat Hand (Idle)"
}

# ==========================================
# INITIALIZATION
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

def get_next_session_dir(base_path="dataset"):
    os.makedirs(base_path, exist_ok=True)
    existing = glob.glob(os.path.join(base_path, "session_*"))
    session_nums = [int(os.path.basename(p).split("_")[1]) for p in existing if os.path.basename(p).split("_")[1].isdigit()]
    next_num = max(session_nums, default=0) + 1
    session_dir = os.path.join(base_path, f"session_{next_num:03d}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir, f"session_{next_num:03d}"

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_palm_normal(coords):
    # Cross product between Wrist->IndexMCP (v1) and Wrist->PinkyMCP (v2)
    v1 = coords[5] - coords[0]
    v2 = coords[17] - coords[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 1e-6 else normal

def extract_features(landmarks, handedness_label):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    # 1. Zero-center relative to wrist (Landmark 0)
    wrist = coords[0]
    centered_coords = coords - wrist
    
    # 2. Scale-invariant divisor (wrist [0] -> middle MCP [9])
    scale_factor = np.linalg.norm(centered_coords[9])
    if scale_factor < 1e-6:
        scale_factor = 1.0
        
    normalized_coords = centered_coords / scale_factor
    flattened_coords = normalized_coords.flatten() # 63 elements
    
    # 3. Key Distances
    d_thumb_index = np.linalg.norm(normalized_coords[4] - normalized_coords[8])
    d_thumb_pinky = np.linalg.norm(normalized_coords[4] - normalized_coords[20])
    d_wrist_thumb = np.linalg.norm(normalized_coords[4])
    d_wrist_index = np.linalg.norm(normalized_coords[8])
    d_wrist_middle = np.linalg.norm(normalized_coords[12])
    d_wrist_ring = np.linalg.norm(normalized_coords[16])
    d_wrist_pinky = np.linalg.norm(normalized_coords[20])
    
    distances = np.array([
        d_thumb_index, d_thumb_pinky, d_wrist_thumb, 
        d_wrist_index, d_wrist_middle, d_wrist_ring, d_wrist_pinky
    ]) # 7 elements
    
    # 4. Finger Curl Angles
    angle_thumb = calculate_angle(normalized_coords[2], normalized_coords[3], normalized_coords[4])
    angle_index = calculate_angle(normalized_coords[5], normalized_coords[6], normalized_coords[8])
    angle_middle = calculate_angle(normalized_coords[9], normalized_coords[10], normalized_coords[12])
    angle_ring = calculate_angle(normalized_coords[13], normalized_coords[14], normalized_coords[16])
    angle_pinky = calculate_angle(normalized_coords[17], normalized_coords[18], normalized_coords[20])
    
    angles = np.array([angle_thumb, angle_index, angle_middle, angle_ring, angle_pinky]) # 5 elements
    
    # 5. Palm Orientation (Vector Normal - 3 elements)
    palm_normal = calculate_palm_normal(normalized_coords)
    
    # Combined feature vector: 63 + 7 + 5 + 3 = 78 elements
    feature_vector = np.hstack([flattened_coords, distances, angles, palm_normal])
    
    # Meta features: Handedness (1 for Right, 0 for Left)
    is_right_hand = 1.0 if handedness_label == "Right" else 0.0
    
    return feature_vector, is_right_hand

def is_duplicate(new_features, last_features, threshold=DUPLICATE_THRESHOLD):
    if last_features is None:
        return False
    # Compare raw normalized landmark coordinates (first 63 elements)
    delta = np.linalg.norm(new_features[:63] - last_features[:63])
    return delta < threshold

def draw_ui_overlay(frame, class_id, count, total, status_msg, recording, paused):
    h, w, _ = frame.shape
    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
    
    cv2.putText(frame, f"Gesture [{class_id}]: {GESTURES[class_id]}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    color = (0, 255, 0) if recording and not paused else ((0, 165, 255) if paused else (0, 0, 255))
    cv2.putText(frame, f"Status: {status_msg}", (15, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Live progress bar at bottom
    progress = int((count / total) * w)
    cv2.rectangle(frame, (0, h - 20), (w, h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h - 20), (progress, h), (0, 255, 0), -1)
    cv2.putText(frame, f"{count}/{total}", (w - 100, h - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def collect_gesture_session(class_id, session_dir, session_id):
    cap = cv2.VideoCapture(0)
    dataset = []
    
    recording = False
    paused = False
    count = 0
    frame_idx = 0
    last_features = None
    countdown_start = None
    
    feature_cols = [f"feat_{i}" for i in range(78)]
    meta_cols = ["is_right_hand", "timestamp", "session_id", "label"]
    all_cols = feature_cols + meta_cols

    while cap.isOpened() and count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame_idx += 1
        
        hand_detected = results.multi_hand_landmarks is not None
        
        # State: Countdown
        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, str(remaining), (frame.shape[1]//2 - 20, frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)
                draw_ui_overlay(frame, class_id, count, SAMPLES_PER_GESTURE, f"Starting in {remaining}s...", False, False)
                cv2.imshow("Edge AI Data Collector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                countdown_start = None
                recording = True

        # State: Pause handling
        if recording and not hand_detected:
            paused = True
            status_msg = "PAUSED (Hand missing!)"
        elif recording and hand_detected:
            paused = False
            status_msg = "RECORDING..."
        elif not recording:
            status_msg = "Press 's' to start countdown"

        # Frame Processing
        if hand_detected:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                handedness_label = handedness.classification[0].label
                
                if recording and not paused and (frame_idx % FRAME_STRIDE == 0):
                    features, is_right = extract_features(hand_landmarks, handedness_label)
                    
                    # Duplicate rejection check
                    if not is_duplicate(features, last_features):
                        timestamp = time.time()
                        row = list(features) + [is_right, timestamp, session_id, class_id]
                        dataset.append(row)
                        last_features = features
                        count += 1
                    
        draw_ui_overlay(frame, class_id, count, SAMPLES_PER_GESTURE, status_msg, recording, paused)
        cv2.imshow("Edge AI Data Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not recording:
            countdown_start = time.time()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Save session CSV
    if dataset:
        df_gesture = pd.DataFrame(dataset, columns=all_cols)
        gesture_csv_path = os.path.join(session_dir, f"gesture_{class_id}.csv")
        df_gesture.to_csv(gesture_csv_path, index=False)
        print(f" [Saved] Class {class_id} saved to {gesture_csv_path} ({len(df_gesture)} rows)")
        return df_gesture
    return pd.DataFrame()

if __name__ == "__main__":
    session_dir, session_id = get_next_session_dir("dataset")
    print("===========================================")
    print(f" Starting Data Collection Session: {session_id}")
    print(" Target: 400 clean frames per gesture class")
    print("===========================================")
    
    session_dataframes = []
    
    for class_id in range(5):
        print(f"\nReady for Class {class_id}: {GESTURES[class_id]}")
        df_g = collect_gesture_session(class_id, session_dir, session_id)
        if not df_g.empty:
            session_dataframes.append(df_g)
            
    # Auto-consolidate to master dataset
    if session_dataframes:
        master_path = os.path.join("dataset", "master_gesture_dataset.csv")
        new_session_df = pd.concat(session_dataframes, ignore_index=False)
        
        if os.path.exists(master_path):
            existing_master = pd.read_csv(master_path)
            combined_master = pd.concat([existing_master, new_session_df], ignore_index=True)
            combined_master.to_csv(master_path, index=False)
            print(f"\n[UPDATED] {master_path} total rows: {len(combined_master)}")
        else:
            new_session_df.to_csv(master_path, index=False)
            print(f"\n[CREATED] {master_path} total rows: {len(new_session_df)}")