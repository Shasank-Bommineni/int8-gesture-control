import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

# Paths
KERAS_MODEL_PATH = os.path.join("models", "baseline_fp32.keras")
TEST_DATA_PATH = os.path.join("models", "test_data.npz")
TFLITE_INT8_PATH = os.path.join("models", "gesture_model_int8.tflite")

# 1. Load Saved Test Data and Baseline Keras Model
if not os.path.exists(TEST_DATA_PATH) or not os.path.exists(KERAS_MODEL_PATH):
    raise FileNotFoundError("Missing baseline model or test data in 'models/' folder. Run Stage 3 first.")

data = np.load(TEST_DATA_PATH)
X_test, y_test = data["X_test"], data["y_test"]

keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
print(f"Loaded FP32 Keras model from: {KERAS_MODEL_PATH}")

# 2. Representative Calibration Dataset Generator
# INT8 quantization needs a sample of real input vectors to calculate tensor dynamic ranges (min/max)
def representative_dataset_gen():
    # Use first 200 samples from test set as calibration input
    for sample in X_test[:200]:
        # Reshape to (1, num_features) with float32 type
        yield [sample.astype(np.float32).reshape(1, -1)]

# 3. Apply Post-Training INT8 Quantization (PTQ)
print("\nApplying Post-Training INT8 Quantization (PTQ)...")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

# Enforce FULL INT8 precision (inputs, outputs, and internal node operations)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

# Save the quantized .tflite flatbuffer binary
with open(TFLITE_INT8_PATH, "wb") as f:
    f.write(tflite_quant_model)

print(f"[SAVED] Quantized INT8 model exported to: {TFLITE_INT8_PATH}")

# 4. Evaluate Quantized INT8 Model on Test Set via TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_INT8_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Extract quantization scaling parameters (S, Z) to quantize FP32 inputs -> INT8
input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

y_pred_int8 = []

for sample in X_test:
    # Scale float input to int8: q = round(v / scale) + zero_point
    quantized_input = np.round(sample / input_scale + input_zero_point).astype(np.int8)
    quantized_input = np.expand_dims(quantized_input, axis=0)

    interpreter.set_tensor(input_details["index"], quantized_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details["index"])
    # Dequantize or directly get argmax class prediction
    predicted_class = np.argmax(output[0])
    y_pred_int8.append(predicted_class)

y_pred_int8 = np.array(y_pred_int8)
int8_acc = accuracy_score(y_test, y_pred_int8)

# 5. Measure Model Size Reduction
fp32_size_kb = os.path.getsize(KERAS_MODEL_PATH) / 1024.0
int8_size_kb = os.path.getsize(TFLITE_INT8_PATH) / 1024.0
size_reduction = (1 - (int8_size_kb / fp32_size_kb)) * 100

print(f"\n==========================================")
print(f" BASELINE VS QUANTIZED COMPARISON")
print(f"==========================================")
print(f" Baseline Model Size (FP32)  : {fp32_size_kb:.2f} KB")
print(f" Quantized Model Size (INT8) : {int8_size_kb:.2f} KB")
print(f" Storage Footprint Reduction : {size_reduction:.2f}% Smaller")
print(f" INT8 Test Set Accuracy      : {int8_acc * 100:.2f}%")
print(f"==========================================")