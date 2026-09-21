import os
import json
import random
from PIL import Image

def setup_directories():
    base_dirs = ["data/raw", "data/processed", "data/generated"]
    sub_dirs = ["vision/mjsynth", "reasoning/coco", "audio/librispeech"]
    for b in base_dirs:
        for s in sub_dirs:
            os.makedirs(os.path.join(b, s), exist_ok=True)

def generate_vision_toy_data():
    raw_dir = "data/raw/vision/mjsynth"
    annotations = []
    for i in range(100):
        filename = f"img_{i}.jpg"
        img = Image.new("RGB", (100, 32), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        img.save(os.path.join(raw_dir, filename))
        label = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))
        annotations.append({"file": filename, "label": label})
    
    with open(os.path.join(raw_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f)

def generate_reasoning_toy_data():
    raw_dir = "data/raw/reasoning/coco"
    annotations = []
    for i in range(100):
        filename = f"vqa_{i}.jpg"
        img = Image.new("RGB", (224, 224), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        img.save(os.path.join(raw_dir, filename))
        annotations.append({
            "file": filename,
            "question": "What color is the image?",
            "answer": "mixed"
        })
    with open(os.path.join(raw_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f)

def generate_audio_toy_data():
    raw_dir = "data/raw/audio/librispeech"
    # Create fake 1-second 16kHz wav files with random noise
    import wave
    import struct
    annotations = []
    for i in range(100):
        filename = f"audio_{i}.wav"
        with wave.open(os.path.join(raw_dir, filename), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            data = [int(random.gauss(0, 5000)) for _ in range(16000)]
            data = [max(-32768, min(32767, x)) for x in data]
            w.writeframes(struct.pack('<' + 'h'*len(data), *data))
        
        annotations.append({"file": filename, "transcript": "fake transcript"})
        
    with open(os.path.join(raw_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f)

if __name__ == "__main__":
    setup_directories()
    generate_vision_toy_data()
    generate_reasoning_toy_data()
    generate_audio_toy_data()
    print("✅ Toy datasets generated successfully.")
