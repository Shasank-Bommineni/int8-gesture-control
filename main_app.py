import os
import cv2
import pickle
import numpy as np
import open3d as o3d
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter  # <-- 1. Added for Prediction Majority Voting

# MediaPipe Solutions setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ==========================================
# 1. LOAD ASSETS (TFLite INT8 + Scaler)
# ==========================================
TFLITE_MODEL_PATH = os.path.join("models", "gesture_model_int8.tflite")
SCALER_PATH = os.path.join("models", "scaler.pkl")

if not os.path.exists(TFLITE_MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Missing models/gesture_model_int8.tflite or models/scaler.pkl. Run Stages 3/4 first.")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]

GESTURE_MAP = {
    0: "Rotate Mode (Open Palm)",
    1: "Grab/Move Mode (Fist)",
    2: "Zoom Mode (Pinch)",
    3: "Select Mode (Point)",
    4: "Idle (Flat Hand)"
}

# ==========================================
# 2. FEATURE EXTRACTION PIPELINE
# ==========================================
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_palm_normal(coords):
    v1 = coords[5] - coords[0]
    v2 = coords[17] - coords[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 1e-6 else normal

def extract_features(landmarks, handedness_label):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    wrist = coords[0]
    centered_coords = coords - wrist
    
    scale_factor = np.linalg.norm(centered_coords[9])
    if scale_factor < 1e-6:
        scale_factor = 1.0
        
    normalized_coords = centered_coords / scale_factor
    flattened_coords = normalized_coords.flatten()
    
    # Key Distances
    d_thumb_index = np.linalg.norm(normalized_coords[4] - normalized_coords[8])
    d_thumb_pinky = np.linalg.norm(normalized_coords[4] - normalized_coords[20])
    d_wrist_thumb = np.linalg.norm(normalized_coords[4])
    d_wrist_index = np.linalg.norm(normalized_coords[8])
    d_wrist_middle = np.linalg.norm(normalized_coords[12])
    d_wrist_ring = np.linalg.norm(normalized_coords[16])
    d_wrist_pinky = np.linalg.norm(normalized_coords[20])
    distances = np.array([d_thumb_index, d_thumb_pinky, d_wrist_thumb, d_wrist_index, d_wrist_middle, d_wrist_ring, d_wrist_pinky])
    
    # Curl Angles
    angle_thumb = calculate_angle(normalized_coords[2], normalized_coords[3], normalized_coords[4])
    angle_index = calculate_angle(normalized_coords[5], normalized_coords[6], normalized_coords[8])
    angle_middle = calculate_angle(normalized_coords[9], normalized_coords[10], normalized_coords[12])
    angle_ring = calculate_angle(normalized_coords[13], normalized_coords[14], normalized_coords[16])
    angle_pinky = calculate_angle(normalized_coords[17], normalized_coords[18], normalized_coords[20])
    angles = np.array([angle_thumb, angle_index, angle_middle, angle_ring, angle_pinky])
    
    # Palm Normal
    palm_normal = calculate_palm_normal(normalized_coords)
    
    feature_vector = np.hstack([flattened_coords, distances, angles, palm_normal])
    is_right = 1.0 if handedness_label == "Right" else 0.0
    
    return np.append(feature_vector, is_right)

def predict_gesture(raw_features):
    scaled_features = scaler.transform(raw_features.reshape(1, -1))[0]
    
    # Quantize Float32 -> INT8
    quantized_input = np.round(scaled_features / input_scale + input_zero_point).astype(np.int8)
    quantized_input = np.expand_dims(quantized_input, axis=0)
    
    interpreter.set_tensor(input_details["index"], quantized_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details["index"])
    return np.argmax(output[0])

# ==========================================
# 3. OPEN3D VIEWPORT INITIALIZATION
# ==========================================
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Edge AI 3D Viewport", width=800, height=600)

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(mesh)

# Tracking & Smoothing Initialization
prev_x, prev_y = None, None
smooth_dx, smooth_dy = 0.0, 0.0
alpha = 0.35  # Smoothing factor: lower = smoother/slower, higher = faster/more reactive

# Prediction Voting Buffer (stores last 5 raw predictions)
prediction_queue = deque(maxlen=5)

# ==========================================
# 4. MAIN REAL-TIME EXECUTION LOOP
# ==========================================
cap = cv2.VideoCapture(0)
print("\n[ACTIVE] Edge AI Gesture Control Running with Motion & Prediction Smoothing.")
print("Press 'q' in OpenCV window to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    current_gesture = 4 # Default: Idle
    gesture_text = GESTURE_MAP[current_gesture]
    
    if results.multi_hand_landmarks and results.multi_handedness:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label
        
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 1. Feature Extraction & Raw Prediction
        raw_feat = extract_features(hand_landmarks, handedness)
        raw_pred = predict_gesture(raw_feat)
        
        # 2. Majority Vote De-Noising (Pick most common gesture over last 5 frames)
        prediction_queue.append(raw_pred)
        current_gesture = Counter(prediction_queue).most_common(1)[0][0]
        gesture_text = GESTURE_MAP[current_gesture]
        
        # 3. Continuous Landmark Delta Tracking (Index Tip: Landmark 8)
        curr_x = hand_landmarks.landmark[8].x
        curr_y = hand_landmarks.landmark[8].y
        
        if prev_x is not None and prev_y is not None:
            raw_dx = (curr_x - prev_x) * 5.0
            raw_dy = (curr_y - prev_y) * 5.0
            
            # Exponential Moving Average (EMA) Low-Pass Filter
            smooth_dx = alpha * raw_dx + (1 - alpha) * smooth_dx
            smooth_dy = alpha * raw_dy + (1 - alpha) * smooth_dy
            
            # Apply smoothed transformations to the 3D mesh
            if current_gesture == 0:  # Rotate Mode
                R = mesh.get_rotation_matrix_from_xyz((smooth_dy, smooth_dx, 0))
                mesh.rotate(R, center=(0, 0, 0))
            elif current_gesture == 1:  # Move Mode
                mesh.translate((smooth_dx * 0.5, -smooth_dy * 0.5, 0))
            elif current_gesture == 2:  # Zoom Mode
                scale_factor = 1.0 + (smooth_dx * 0.2)
                if 0.5 < scale_factor < 2.0:
                    mesh.scale(scale_factor, center=mesh.get_center())
                    
        prev_x, prev_y = curr_x, curr_y
    else:
        prev_x, prev_y = None, None
        smooth_dx, smooth_dy = 0.0, 0.0
        prediction_queue.clear()

    # Render Open3D Scene
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    
    # Render Overlay
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, f"Active Mode: {gesture_text}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Webcam Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()