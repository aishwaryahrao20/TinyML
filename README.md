# Audio-Based Keyword Spotting System Using TinyML

A real-time, on-device keyword spotter that recognizes five spoken commands — **all, must, none, never, only** — from a continuous microphone stream on an **Arduino Nano 33 BLE Sense**. No cloud connectivity, no host compute — every inference decision is made locally in milliseconds.

> **Course:** Embedded Systems — Arizona State University  
> **Duration:** Aug – Dec 2025  
> **Author:** [Aishwarya Hareesh Rao](https://aishwaryarao-portfolio.vercel.app/)

---

## Overview

This project covers the full TinyML development cycle for audio classification: data collection, feature engineering, model training, and embedded deployment — all targeting a single resource-constrained microcontroller. The system captures 1-second audio clips from the onboard PDM microphone, extracts lightweight log-energy features, and classifies them through a compact MLP deployed as a C header file with zero runtime dependencies.

---

## Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│   Record    │     │  Feature Extract │     │   Train MLP     │     │  Deploy to MCU       │
│  1s WAV @   │────►│  101 frames ×    │────►│  Flatten → 64   │────►│  Export as C header  │
│  16 kHz     │     │  13 log-energy   │     │  → 64 → 5       │     │  (keyword_linear_    │
│             │     │  dims            │     │  (Softmax)      │     │   model.h)           │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────────┘
```

### 1. Data Collection

Audio clips were recorded as **1-second WAV files at 16 kHz**, organized into keyword-named folders for automatic label inference. Five keywords were chosen: *all*, *must*, *none*, *never*, and *only*.

### 2. Feature Extraction

Each clip is segmented into **101 frames** and converted to a log-energy feature — `log(1 + mean|x|)` per frame — replicated across **13 dimensions** to produce a **(101 × 13)** input tensor. Log-energy was chosen over MFCC because it cuts per-frame compute by ~80 % while retaining enough spectral envelope for a five-class vocabulary.

### 3. Training & Evaluation

The classifier uses a three-layer MLP optimized for minimal FLOP count:

```
Input (101 × 13)
  │
  ├── Flatten
  ├── Dense(64, ReLU)
  ├── Dense(64, ReLU)
  └── Dense(5, Softmax)
```

The model was evaluated with per-class precision, recall, and F1-score.

### 4. Embedded Deployment

Model weights are exported as a **C header file** (`keyword_linear_model.h`) with training-set normalization statistics (mean, std) embedded alongside — no TFLite interpreter or runtime library is needed. The Arduino sketch reads audio from the PDM microphone, computes features in-place, runs a forward pass through the weight matrices, and outputs the predicted keyword.

---

## Engineering Decisions

- **Log-energy over MFCC.** Cuts per-frame compute by ~80 % with no meaningful accuracy loss for a small vocabulary. Practical for a Cortex-M4 running at 64 MHz.
- **C header deployment (no TFLite).** A single model with fixed architecture fits cleanly into a compile-time C header. This avoids the memory overhead of a TFLite interpreter and keeps the firmware footprint minimal.
- **Normalization parity.** Training-set mean and standard deviation are exported and hard-coded on the device. Even small drift between training and inference normalization silently degrades accuracy — exporting them explicitly eliminates that risk.

---

## Hardware

| Component | Details |
|-----------|---------|
| Board     | Arduino Nano 33 BLE Sense (Cortex-M4F @ 64 MHz, 256 KB RAM) |
| Microphone| Onboard PDM digital microphone |
| Interface | USB serial for monitoring predictions |

## Software Stack

| Layer | Tools |
|-------|-------|
| Data collection | Python, WAV recording scripts |
| Feature extraction | NumPy, custom log-energy pipeline |
| Training | TensorFlow / Keras, scikit-learn |
| Export | Custom Python → C header converter |
| Firmware | Arduino IDE, C++ |

## Skills

`TinyML` · `Embedded AI` · `Python` · `TensorFlow / Keras` · `Arduino` · `C++` · `Signal Processing` · `Feature Extraction` · `Model Optimization`

---

## Keywords

| Class | Keyword |
|-------|---------|
| 0     | all     |
| 1     | must    |
| 2     | none    |
| 3     | never   |
| 4     | only    |

---

## License

This project was developed for academic coursework at Arizona State University.

## Contact

**Aishwarya Hareesh Rao**  
[Portfolio](https://aishwaryarao-portfolio.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/aishwaryahrao) · [GitHub](https://github.com/aishwaryahrao20)
