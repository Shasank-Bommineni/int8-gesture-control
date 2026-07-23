import os
import sys
import time
import psutil
import numpy as np
import tensorflow as tf

print("Starting benchmark script execution...", flush=True)

# File Paths
KERAS_MODEL_PATH = os.path.join("models", "baseline_fp32.keras")
TFLITE_INT8_PATH = os.path.join("models", "gesture_model_int8.tflite")
TEST_DATA_PATH = os.path.join("models", "test_data.npz")

# Verify required files exist
for path in [KERAS_MODEL_PATH, TFLITE_INT8_PATH, TEST_DATA_PATH]:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}", flush=True)
        sys.exit(1)

data = np.load(TEST_DATA_PATH)
X_test = data["X_test"]
num_samples = len(X_test)
print(f"Loaded test dataset with {num_samples} samples.", flush=True)

def profile_keras_fp32(model_path, samples, iterations=500):
    model = tf.keras.models.load_model(model_path)
    # Warmup run
    _ = model(samples[:1], training=False)
    
    latencies = []
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    start_total = time.perf_counter()
    for i in range(iterations):
        sample = samples[i % num_samples : (i % num_samples) + 1]
        t0 = time.perf_counter_ns()
        _ = model(sample, training=False)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)
        
    end_total = time.perf_counter()
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    fps = iterations / (end_total - start_total)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    peak_mem = max(mem_before, mem_after)
    
    return p50, p99, fps, peak_mem

def profile_tflite_int8(model_path, samples, iterations=500):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    
    # Warmup
    dummy = np.zeros(input_details["shape"], dtype=np.int8)
    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()
    
    latencies = []
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    start_total = time.perf_counter()
    for i in range(iterations):
        sample = samples[i % num_samples]
        quantized = np.round(sample / input_scale + input_zero_point).astype(np.int8)
        quantized = np.expand_dims(quantized, axis=0)
        
        t0 = time.perf_counter_ns()
        interpreter.set_tensor(input_details["index"], quantized)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)
        
    end_total = time.perf_counter()
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    fps = iterations / (end_total - start_total)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    peak_mem = max(mem_before, mem_after)
    
    return p50, p99, fps, peak_mem

print("\n[1/2] Benchmarking Baseline FP32 Model...", flush=True)
fp32_p50, fp32_p99, fp32_fps, fp32_mem = profile_keras_fp32(KERAS_MODEL_PATH, X_test)

print("[2/2] Benchmarking Quantized INT8 Model...", flush=True)
int8_p50, int8_p99, int8_fps, int8_mem = profile_tflite_int8(TFLITE_INT8_PATH, X_test)

fp32_size = os.path.getsize(KERAS_MODEL_PATH) / 1024.0
int8_size = os.path.getsize(TFLITE_INT8_PATH) / 1024.0

print("\n" + "="*60, flush=True)
print(" FINAL EDGE DEPLOYMENT BENCHMARK SUMMARY TABLE", flush=True)
print("="*60, flush=True)
print(f"{'Metric':<25} | {'FP32 Baseline':<15} | {'INT8 Quantized':<15}", flush=True)
print("-" * 60, flush=True)
print(f"{'Binary Size (KB)':<25} | {fp32_size:<15.2f} | {int8_size:<15.2f}", flush=True)
print(f"{'Median Latency p50 (ms)':<25} | {fp32_p50:<15.3f} | {int8_p50:<15.3f}", flush=True)
print(f"{'Tail Latency p99 (ms)':<25} | {fp32_p99:<15.3f} | {int8_p99:<15.3f}", flush=True)
print(f"{'Throughput (FPS)':<25} | {fp32_fps:<15.1f} | {int8_fps:<15.1f}", flush=True)
print(f"{'Peak Memory (MB)':<25} | {fp32_mem:<15.2f} | {int8_mem:<15.2f}", flush=True)
print("="*60, flush=True)