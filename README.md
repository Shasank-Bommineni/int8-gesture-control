# Edge AI Gesture Control & Compression Pipeline (TinyML)

An end-to-end TinyML application that extracts hand geometry features, trains an optimized gesture classification neural network, applies INT8 post-training quantization, and deploys a sub-13 KB model to control a real-time interactive 3D viewport.

---

## Quantization & Edge Deployment Benchmarks

Evaluated on an isolated CPU/RAM-constrained runtime environment (`--cpus=0.5`, `--memory=256m`):

| Metric | FP32 Keras Baseline | INT8 Quantized TFLite | Delta |
| :--- | :--- | :--- | :--- |
| Model binary size | 127.82 KB | **12.84 KB** | 89.95% smaller |
| Median latency (p50) | 5.632 ms | **0.005 ms** | ~1,127x faster |
| Tail latency (p99) | 11.858 ms | **0.030 ms** | ~395x faster |
| Throughput | 158.9 FPS | **45,760 FPS** | — |
| Peak memory usage | 338.83 MB | 340.29 MB | stable |

**Methodology note:** the FP32 figures measure `model.predict()` on the full Keras/TensorFlow runtime, which carries fixed per-call graph-dispatch overhead in Python. The INT8 figures measure the raw `Interpreter.invoke()` call on the compiled TFLite runtime, which has minimal dispatch overhead by design. Part of the latency delta reflects this runtime/API difference, not quantization alone — INT8 quantization's direct contribution is the ~10x size reduction and the arithmetic speedup from int8 vs float32 ops. Both effects are real and both matter for edge deployment, but they're reported separately here for accuracy rather than conflated into one headline number.

---

## System Architecture & Pipeline

```
[ Webcam Feed ]
      |
      v
[ MediaPipe Hand Detection ]
      |
      v
[ 79-Element Geometric Feature Extractor ]
  - Wrist-centered normalized landmark coordinates (63)
  - Inter-landmark Euclidean distances (7)
  - Joint curl angles (5)
  - Palm normal vector (3)
  - Handedness flag (1)
      |
      v
[ StandardScaler ] --> [ INT8 Quantized TFLite Model, 12.8 KB ]
                                |
                                v
                 [ Gesture Classification Output ]
                                |
                                v
                 [ Real-Time Open3D Viewport ]
```

**Gesture-to-action mapping:**

| Gesture | Viewport Action |
| :--- | :--- |
| Open Palm | Rotate mode |
| Fist | Grab / Move mode |
| Pinch | Zoom / Scale mode |
| Point | Select mode |
| Flat Hand (static) | Idle |

---

## Quickstart & Reproduction

### 1. Prerequisites & Environment Setup

```bash
git clone https://github.com/your-username/edge-ai-gesture-control.git
cd edge-ai-gesture-control

# Create and activate virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Custom Gesture Data (optional — pretrained data included)

Records real-time 79-element hand feature vectors via webcam, labeled by gesture class:

```bash
python collect_data.py
```

### 3. Train the Baseline FP32 Model

Trains the Dense Keras classifier and exports the feature scaler (`scaler.pkl`):

```bash
python train_baseline_v2.py
```

### 4. Quantize to INT8 TFLite

Runs post-training quantization to compress the model to sub-13 KB:

```bash
python quantize_tflite.py
```

### 5. Run Edge Benchmarks

Evaluates latency (p50/p99), throughput (FPS), memory footprint, and binary size:

```bash
python benchmark_edge.py
```

### 6. Launch the Real-Time 3D Viewport

Runs the end-to-end application, controlling the Open3D viewport with live hand gestures:

```bash
python main_app.py
```

Press `q` in the OpenCV webcam window to exit.

---

## Dockerized Benchmarking (optional)

Profile the INT8 model inside an isolated, resource-constrained execution container:

```bash
# Build the image
docker build -t tinyml-benchmark .

# Run the benchmark under CPU/RAM throttling
docker run --rm --cpus="0.5" --memory="256m" tinyml-benchmark python -u benchmark_edge.py
```

---

## Notes

- All benchmarks are reproducible via `benchmark_edge.py`; raw output is not hand-edited.
- The feature extractor is shared verbatim across data collection, training, and inference (`feature_extraction.py`) to guarantee train/serve consistency.
- This project intentionally simulates edge-device constraints via Docker CPU/RAM limits rather than claiming deployment to physical embedded hardware, since no such hardware was used.