import os
import librosa
import soundfile as sf

# where your .ogg data is
INPUT_ROOT = "ogg_input"
# where your main .wav dataset lives
OUTPUT_ROOT = "data_local"

TARGET_SR = 16000

def convert_experiment(exp_name):
    in_base = os.path.join(INPUT_ROOT, exp_name, "audio data wav")
    out_base = os.path.join(OUTPUT_ROOT, exp_name, "audio data wav")

    if not os.path.isdir(in_base):
        print(f"[WARN] No input experiment dir: {in_base}")
        return

    labels = os.listdir(in_base)
    for label in labels:
        in_dir = os.path.join(in_base, label)
        if not os.path.isdir(in_dir):
            continue
        out_dir = os.path.join(out_base, label)
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(in_dir):
            if not fname.lower().endswith(".ogg"):
                continue
            in_path = os.path.join(in_dir, fname)
            y, _ = librosa.load(in_path, sr=TARGET_SR, mono=True)
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, base + ".wav")
            sf.write(out_path, y, TARGET_SR)
            print("Converted:", in_path, "->", out_path)

if __name__ == "__main__":
    # change or extend this list to your experiments
    for exp in ["experiment5"]:
        convert_experiment(exp)