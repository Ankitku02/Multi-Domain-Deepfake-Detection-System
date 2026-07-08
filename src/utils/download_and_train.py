import os
import sys
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.preprocess import compute_frequency_maps
from src.training import train as train_module

def main():
    print("==================================================================")
    print("   REAL DATASET PREPARATION & MODEL TRAINING RUN   ")
    print("==================================================================")
    
    # 1. Path configuration
    preprocessed_dir = os.path.join(project_root, "data/preprocessed")
    faces_out_dir = os.path.join(preprocessed_dir, "faces")
    freq_out_dir = os.path.join(preprocessed_dir, "frequency")
    metadata_csv = os.path.join(project_root, "data/metadata.csv")
    
    os.makedirs(faces_out_dir, exist_ok=True)
    os.makedirs(freq_out_dir, exist_ok=True)
    
    metadata_records = []
    max_per_class = 50
    counts = {0: 0, 1: 0} # 0: Real, 1: Fake
    
    # Check if preprocessed FaceForensics++ files are already on disk
    local_samples_found = True
    for label in [0, 1]:
        for i in range(max_per_class):
            v_name = f"ffpp_sample_{label}_{i}"
            face_img_name = f"{v_name}_frame_0.jpg"
            face_img_path = os.path.join(faces_out_dir, face_img_name)
            freq_npy_name = f"{v_name}_frame_0.npy"
            freq_npy_path = os.path.join(freq_out_dir, freq_npy_name)
            
            if not (os.path.exists(face_img_path) and os.path.exists(freq_npy_path)):
                local_samples_found = False
                break
        if not local_samples_found:
            break
            
    if local_samples_found:
        print("Found 100 face images and frequency maps locally on disk (50 real, 50 fake).")
        print("Generating metadata.csv and training without downloading...")
        for label in [0, 1]:
            for i in range(max_per_class):
                v_name = f"ffpp_sample_{label}_{i}"
                face_img_path = os.path.join(faces_out_dir, f"{v_name}_frame_0.jpg")
                freq_npy_path = os.path.join(freq_out_dir, f"{v_name}_frame_0.npy")
                
                # Split logic (70% train, 14% val, 16% test)
                if i < 35:
                    split = "train"
                elif i < 42:
                    split = "val"
                else:
                    split = "test"
                    
                metadata_records.append({
                    "video_name": v_name,
                    "frame_idx": 0,
                    "face_path": os.path.relpath(face_img_path, project_root),
                    "freq_path": os.path.relpath(freq_npy_path, project_root),
                    "label": label,
                    "split": split
                })
    else:
        # Fallback to streaming from Hugging Face
        print("Preprocessed files not fully found on disk. Connecting to Hugging Face dataset stream: TsienDragon/ffplusplus_c23_frames...")
        try:
            dataset = load_dataset("TsienDragon/ffplusplus_c23_frames", split="train", streaming=True)
        except Exception as e:
            print(f"Error loading dataset from HF: {e}")
            return
            
        print("Successfully connected to dataset stream! Extracting files...")
        idx = 0
        try:
            for item in dataset:
                raw_label = item.get("label")
                if isinstance(raw_label, str):
                    raw_label_lower = raw_label.lower()
                    if raw_label_lower == 'real':
                        label = 0
                    elif raw_label_lower == 'deepfake':
                        label = 1
                    else:
                        continue
                elif isinstance(raw_label, (int, float)):
                    label = int(raw_label)
                else:
                    continue

                # Ensure label is 0 or 1
                if label not in [0, 1]:
                    continue
                    
                if counts[label] >= max_per_class:
                    if all(c >= max_per_class for c in counts.values()):
                        break
                    continue
                    
                img = item.get("image")
                if img is None:
                    continue
                    
                # Convert PIL image to BGR for saving
                img_rgb = np.array(img.convert("RGB"))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img_bgr = cv2.resize(img_bgr, (224, 224))
                
                # Save face crop image
                v_name = f"ffpp_sample_{label}_{counts[label]}"
                face_img_name = f"{v_name}_frame_0.jpg"
                face_img_path = os.path.join(faces_out_dir, face_img_name)
                cv2.imwrite(face_img_path, img_bgr)
                
                # Compute log log-magnitude maps (FFT/DCT)
                freq_maps = compute_frequency_maps(img_bgr)
                freq_npy_name = f"{v_name}_frame_0.npy"
                freq_npy_path = os.path.join(freq_out_dir, freq_npy_name)
                np.save(freq_npy_path, freq_maps)
                
                # Divide into splits (70% train, 14% val, 16% test)
                if counts[label] < 35:
                    split = "train"
                elif counts[label] < 42:
                    split = "val"
                else:
                    split = "test"
                    
                metadata_records.append({
                    "video_name": v_name,
                    "frame_idx": 0,
                    "face_path": os.path.relpath(face_img_path, project_root),
                    "freq_path": os.path.relpath(freq_npy_path, project_root),
                    "label": label,
                    "split": split
                })
                
                counts[label] += 1
                idx += 1
                
                if idx % 10 == 0:
                    print(f"--> Extracted {idx} images... (Real: {counts[0]}, Fake: {counts[1]})")
        except Exception as iter_err:
            print(f"Warning: Stream iteration interrupted: {iter_err}")
            
    # Save metadata CSV
    df = pd.DataFrame(metadata_records)
    if df.empty:
        print("Error: No samples processed.")
        return
        
    df.to_csv(metadata_csv, index=False)
    print(f"\nSuccess: Prepared {len(df)} images.")
    print(f"Metadata CSV saved to: {metadata_csv}")
    print(f"Split breakdown:\n{df['split'].value_counts()}")
    
    # 3. Train the model on the prepared dataset
    print("\nStarting model training on the dataset...")
    # Train for 3 epochs on the real data
    train_module.main(epochs_override=3)
    
    print("\n==================================================================")
    print("   DATASET TRAINING CYCLE COMPLETED SUCCESSFULLY!   ")
    print("==================================================================")

if __name__ == "__main__":
    main()
