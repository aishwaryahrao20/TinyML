#include <Arduino.h>
#include <PDM.h>
#include <math.h>
#include "keyword_linear_model.h"

// ==================================================
// OPTIONAL DEBUG / EXPERIMENT SWITCHES (Set for ACCURACY)
// ==================================================
// 1 = use the transient-based gate (MUST be 1 for max noise rejection)
#define USE_TRANSIENT_GATE 0

// 1 = print an estimated SNR in dB (for debugging only, no gating)
#define ENABLE_SNR_LOGGING 1 

// --------------------------------------------------
// Audio / feature settings
// --------------------------------------------------
constexpr int   kAudioSampleFrequency = 16000;
constexpr int   kClipSamples          = 16000;
constexpr float kTargetRms            = 0.10f; 

// Order MUST match KEYWORDS list in Colab
static const char* kClassNames[kNumClasses] = {
  "all", "never", "none", "only", "must"
};

// Detection thresholds (Optimized balance)
const float kSilenceThreshold   = 0.001f;
const float kMinP               = 0.08f;
const float kTransientRatio     = 2.0f;     // Ratio for VAD check (spike must be 2x avg noise)
const int   kTransientFrames    = 8;        // Window size for VAD check

// --------------------------------------------------
// Global audio buffer
// --------------------------------------------------
static volatile int16_t g_audio_buffer[kClipSamples];
static volatile int     g_audio_index = 0;
static volatile bool    g_buffer_full = false;
static const int kPdmBufferSamples = 512;
static int16_t   g_pdm_chunk[kPdmBufferSamples];

// --------------------------------------------------
// Feature / network buffers
// --------------------------------------------------
static float g_features[kTimeSteps][kNumFeatures];
static float g_flat[kFlatDim];
static float g_hidden1[kHidden1Dim];
static float g_hidden2[kHidden2Dim];
static float g_logits[kNumClasses];

// EMA smoothing for probabilities
static float g_prob_ema[kNumClasses] = {0.0f};
const float  kEmaAlpha = 0.85f;

// Small bias correction
static const float kBiasCorrect[kNumClasses] = {
  +0.05f,   // all
  -0.03f,   // never
  +0.05f,   // none
  +0.05f,   // only
  -0.03f    // must
};


// ==================================================
// PDM callback (Unchanged)
// ==================================================
void onPDMData() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable <= 0) return;
  int maxBytes    = kPdmBufferSamples * (int)sizeof(int16_t);
  int bytesToRead = (bytesAvailable > maxBytes) ? maxBytes : bytesAvailable;
  int bytesRead = PDM.read(g_pdm_chunk, bytesToRead);
  if (bytesRead <= 0) return;
  int samplesRead = bytesRead / 2;
  int idx = g_audio_index;
  for (int i = 0; i < samplesRead; ++i) {
    if (idx < kClipSamples) {
      g_audio_buffer[idx++] = g_pdm_chunk[i];
    }
  }
  g_audio_index = idx;
  if (g_audio_index >= kClipSamples) {
    g_buffer_full = true;
  }
}

// ==================================================
// Setup PDM mic
// ==================================================
void setupPDM() {
  PDM.onReceive(onPDMData);
  if (!PDM.begin(1, kAudioSampleFrequency)) {
    Serial.println("Failed to start PDM!");
    while (1) { delay(10); }
  }

  // ADJUSTMENT: Gain reduced to 75 for better noise immunity (SNR)
  PDM.setGain(75); 
  Serial.println("PDM Gain set to 75 (Reduced from 85 for better SNR).");
}

// ==================================================
// Preprocess raw mic buffer 
// ==================================================
void PreprocessAudioBuffer() {
  // 1) DC offset removal
  long long sum = 0;
  for (int i = 0; i < kClipSamples; ++i) sum += g_audio_buffer[i];
  float mean = (float)sum / (float)kClipSamples;
  for (int i = 0; i < kClipSamples; ++i) {
    float v = (float)g_audio_buffer[i] - mean;
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    g_audio_buffer[i] = (int16_t)v;
  }
  Serial.print("DC offset removed, mean was: ");
  Serial.println(mean, 2);

  // 2) High-pass filter (0.997f)
  const float hp_alpha = 0.997f;
  float prev_x = (float)g_audio_buffer[0];
  float prev_y = 0.0f;
  for (int i = 0; i < kClipSamples; ++i) {
    float x = (float)g_audio_buffer[i];
    float y = hp_alpha * (prev_y + x - prev_x);
    prev_x = x;
    prev_y = y;
    if (y > 32767.0f) y = 32767.0f;
    if (y < -32768.0f) y = -32768.0f;
    g_audio_buffer[i] = (int16_t)y;
  }

  // 3) Low-pass filter (0.80f)
  const float lp_alpha = 0.80f; // Stable value
  float prev = (float)g_audio_buffer[0];
  for (int i = 1; i < kClipSamples; ++i) {
    float x = (float)g_audio_buffer[i];
    float y = lp_alpha * prev + (1.0f - lp_alpha) * x;
    prev = y;
    if (y > 32767.0f) y = 32767.0f;
    if (y < -32768.0f) y = -32768.0f;
    g_audio_buffer[i] = (int16_t)y;
  }

  // 4) Noise gate (fixed threshold = 400)
  const int16_t gateThreshold = 400;
  for (int i = 0; i < kClipSamples; ++i) {
    int16_t s = g_audio_buffer[i];
    if (s > -gateThreshold && s < gateThreshold) {
      g_audio_buffer[i] = 0;
    }
  }
}

// ==================================================
// Simple silence detector (Unchanged)
// ==================================================
bool isClipSilent() {
  float sumAbs = 0.0f;
  for (int i = 0; i < kClipSamples; ++i) {
    float s = (float)g_audio_buffer[i] / 32768.0f;
    sumAbs += fabsf(s);
  }
  float avg = sumAbs / (float)kClipSamples;
  Serial.print("Clip average amplitude (0-1.0): ");
  Serial.println(avg, 5);
  return (avg < kSilenceThreshold);
}

// ==================================================
// Feature extraction: RMS normalize + log1p(mean|s|)
// (Unchanged)
// ==================================================
float ComputeFeaturesFromGlobalAudio() {
  // 1) Global RMS
  float sumSq = 0.0f;
  for (int i = 0; i < kClipSamples; ++i) {
    float s = (float)g_audio_buffer[i] / 32768.0f;
    sumSq  += s * s;
  }
  float rms = sqrtf(sumSq / (float)kClipSamples);
  float scale = 1.0f;
  if (rms > 1e-6f) {
    scale = kTargetRms / rms;
  }

  // Loosened clamp, max boost set to 15x
  if (scale > 15.0f) scale = 15.0f; 
  if (scale < 0.5f)  scale = 0.5f;

  Serial.print("Clip RMS: ");
  Serial.print(rms, 5);
  Serial.print("  -> scale factor: ");
  Serial.println(scale, 3);

  // 2) Frame-based log energy
  const int samplesPerStep = kClipSamples / kTimeSteps;
  for (int t = 0; t < kTimeSteps; ++t) {
    float sumAbs = 0.0f;
    int start = t * samplesPerStep;
    int end   = start + samplesPerStep;
    if (end > kClipSamples) end = kClipSamples;
    for (int i = start; i < end; ++i) {
      float s = (float)g_audio_buffer[i] / 32768.0f;
      s *= scale;
      sumAbs += fabsf(s);
    }
    float avg   = 0.0f;
    int   count = end - start;
    if (count > 0) {
      avg = sumAbs / (float)count;
    }
    float feature = logf(1.0f + avg);
    for (int c = 0; c < kNumFeatures; ++c) {
      g_features[t][c] = feature;
    }
  }
  return scale;
}

// ==================================================
// Speech Transient Detector (Refined for Accuracy)
// ==================================================
void ClearEmaBuffer() {
    for (int i = 0; i < kNumClasses; ++i) {
        g_prob_ema[i] = 0.0f;
    }
}

bool isNoiseTransientPresent(float scaleFactor) {
  const int   samplesPerStep = kClipSamples / kTimeSteps;
  
  // 1. Calculate energy for kTransientFrames (the first 8 frames)
  for (int t = 0; t < kTransientFrames; ++t) {
    float sumSq = 0.0f;
    int start = t * samplesPerStep;
    int end   = start + samplesPerStep;
    if (end > kClipSamples) end = kClipSamples;
    int count = end - start;
    for (int i = start; i < end; ++i) {
      float s = (float)g_audio_buffer[i] / 32768.0f;
      s *= scaleFactor; 
      sumSq += s * s;
    }
    g_features[t][0] = sumSq / (float)count; 
  }
  
  // 2. Calculate average energy of the "pre-speech" period (first half, 4 frames)
  // This gives a more localized estimate of the noise floor just before the word starts.
  float sumPreEnergy = 0.0f;
  for (int t = 0; t < kTransientFrames / 2; ++t) { 
      sumPreEnergy += g_features[t][0];
  }
  float avgPreEnergy = sumPreEnergy / (float)(kTransientFrames / 2);

  // 3. Look for a spike (transient) in the entire window relative to the pre-energy.
  for (int t = 0; t < kTransientFrames; ++t) {
    if (g_features[t][0] > kTransientRatio * avgPreEnergy) { 
      Serial.print("Transient detected (Spike: ");
      Serial.print(g_features[t][0], 5);
      Serial.print(" > AvgPre*");
      Serial.print(kTransientRatio, 1);
      Serial.print(": ");
      Serial.print(kTransientRatio * avgPreEnergy, 5);
      Serial.println(")");
      return true; 
    }
  }

  Serial.print("No transient detected (AvgPre Energy: ");
  Serial.print(avgPreEnergy, 5);
  Serial.println("). Likely continuous noise.");
  return false; 
}

// ==================================================
// OPTIONAL: SNR estimation (for logging only)
// ==================================================
float EstimateSNRdBFromBuffer() {
#if ENABLE_SNR_LOGGING
  // We work on the preprocessed buffer (same one used for features)
  const int samplesPerStep = kClipSamples / kTimeSteps;
  float frameRms[kTimeSteps];

  for (int t = 0; t < kTimeSteps; ++t) {
    float sumSq = 0.0f;
    int start = t * samplesPerStep;
    int end   = start + samplesPerStep;
    if (end > kClipSamples) end = kClipSamples;
    int count = end - start;
    if (count <= 0) { frameRms[t] = 0.0f; continue; }
    for (int i = start; i < end; ++i) {
      float s = (float)g_audio_buffer[i] / 32768.0f;
      sumSq += s * s;
    }
    frameRms[t] = sqrtf(sumSq / (float)count);
  }

  float temp[kTimeSteps];
  for (int i = 0; i < kTimeSteps; ++i) temp[i] = frameRms[i];

  // Simple insertion sort
  for (int i = 1; i < kTimeSteps; ++i) {
    float key = temp[i];
    int j = i - 1;
    while (j >= 0 && temp[j] > key) {
      temp[j + 1] = temp[j];
      j--;
    }
    temp[j + 1] = key;
  }

  int quarter = kTimeSteps / 4;
  if (quarter < 1) quarter = 1;

  // Noise estimate: average of the quietest 25% of frames
  float noiseSum = 0.0f;
  for (int i = 0; i < quarter; ++i) {
    noiseSum += temp[i];
  }
  float noiseRms = noiseSum / (float)quarter;

  // Signal estimate: average of the loudest 25% of frames
  float signalSum = 0.0f;
  for (int i = kTimeSteps - quarter; i < kTimeSteps; ++i) {
    signalSum += temp[i];
  }
  float signalRms = signalSum / (float)quarter;

  if (noiseRms < 1e-6f) noiseRms = 1e-6f;
  if (signalRms < noiseRms) signalRms = noiseRms;

  float snrLinear = signalRms / noiseRms;
  float snrDb = 20.0f * log10f(snrLinear);

  Serial.print("Estimated SNR (dB): ");
  Serial.println(snrDb, 2);

  return snrDb;
#else
  return 0.0f;
#endif
}

// ==================================================
// Network Forwarding Functions (Unchanged)
// ==================================================
void FlattenAndNormalize() {
  int idx = 0;
  for (int t = 0; t < kTimeSteps; ++t) {
    for (int c = 0; c < kNumFeatures; ++c) {
      float v = g_features[t][c];
      v = (v - kFeatMean[c]) / kFeatStd[c];
      g_flat[idx++] = v;
    }
  }
}

void Dense1Forward() {
  for (int j = 0; j < kHidden1Dim; ++j) {
    float acc = kB1[j];
    for (int i = 0; i < kFlatDim; ++i) {
      acc += g_flat[i] * kW1[j][i];
    }
    g_hidden1[j] = (acc > 0.0f) ? acc : 0.0f;
  }
}

void Dense2Forward() {
  for (int j = 0; j < kHidden2Dim; ++j) {
    float acc = kB2[j];
    for (int i = 0; i < kHidden1Dim; ++i) {
      acc += g_hidden1[i] * kW2[j][i];
    }
    g_hidden2[j] = (acc > 0.0f) ? acc : 0.0f;
  }
}

void Dense3Forward() {
  for (int j = 0; j < kNumClasses; ++j) {
    float acc = kB3[j];
    for (int i = 0; i < kHidden2Dim; ++i) {
      acc += g_hidden2[i] * kW3[j][i];
    }
    g_logits[j] = acc;
  }
}

void Softmax(float* logits, int n) {
  float maxLogit = logits[0];
  for (int i = 1; i < n; ++i) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  float sumExp = 0.0f;
  for (int i = 0; i < n; ++i) {
    logits[i] = expf(logits[i] - maxLogit);
    sumExp += logits[i];
  }

  if (sumExp <= 1e-8f) {
    for (int i = 0; i < n; ++i) logits[i] = 1.0f / (float)n;
  } else {
    for (int i = 0; i < n; ++i) logits[i] /= sumExp;
  }
}

// ==================================================
// Single end-to-end run on 1 second of audio
// ==================================================
void runKeywordSpottingOnce() {
  PreprocessAudioBuffer();

  if (isClipSilent()) {
    Serial.println("-> SILENT / background. Skipping inference.\n");
    return;
  }

  float scaleFactor = ComputeFeaturesFromGlobalAudio();

  // Log SNR for debugging without changing the decision logic
  EstimateSNRdBFromBuffer();

#if USE_TRANSIENT_GATE
  if (!isNoiseTransientPresent(scaleFactor)) {
    Serial.println("-> LOUD, BUT NO SPEECH TRANSIENT. Treating as noise. Skipping inference.\n");
    return;
  }
#endif

  FlattenAndNormalize();
  Dense1Forward();
  Dense2Forward();
  Dense3Forward();
  Softmax(g_logits, kNumClasses);

  // EMA smoothing + bias correction
  for (int i = 0; i < kNumClasses; ++i) {
    g_prob_ema[i] = kEmaAlpha * g_prob_ema[i] + (1.0f - kEmaAlpha) * g_logits[i];
    g_prob_ema[i] += kBiasCorrect[i];
    if (g_prob_ema[i] < 0.0f) g_prob_ema[i] = 0.0f;
  }

  int   bestIdx = 0;
  float bestP   = g_prob_ema[0];

  Serial.print("Probabilities (EMA): ");
  for (int i = 0; i < kNumClasses; ++i) {
    Serial.print(kClassNames[i]);
    Serial.print("=");
    Serial.print(g_prob_ema[i], 4);
    Serial.print("  ");
    if (g_prob_ema[i] > bestP) {
      bestP = g_prob_ema[i];
      bestIdx = i;
    }
  }
  Serial.println();

  if (bestP < kMinP) {
    Serial.print("-> No confident keyword (max p=");
    Serial.print(bestP, 3);
    Serial.println(")\n");
    return;
  }

  // Clear EMA buffer on successful detection
  ClearEmaBuffer();
  
  Serial.print("-> Predicted: ");
  Serial.print(kClassNames[bestIdx]);
  Serial.print("  (p=");
  Serial.print(bestP, 3);
  Serial.println(")\n");
}

// ==================================================
// Arduino setup / loop (Unchanged)
// ==================================================
void setup() {
  Serial.begin(115200); 
  while (!Serial) {;}

  Serial.println("Keyword spotting demo (all/never/none/only/must)");
  Serial.println("Init PDM microphone...");
  setupPDM();

  g_audio_index = 0;
  g_buffer_full = false;
}

void loop() {
  if (!g_buffer_full) {
    delay(5);
    return;
  }

  noInterrupts();
  g_buffer_full = false;
  g_audio_index = 0;
  interrupts();

  Serial.println("---- 1 second audio captured ----");
  runKeywordSpottingOnce();
}