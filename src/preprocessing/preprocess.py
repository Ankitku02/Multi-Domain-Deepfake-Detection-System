import os
import cv2
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .extract_frames import extract_video_frames, detect_and_crop_face
except ImportError:
    try:
        from extract_frames import extract_video_frames, detect_and_crop_face
    except ImportError:
        from src.preprocessing.extract_frames import extract_video_frames, detect_and_crop_face

def compute_dct2d(img):
    """Computes the 2D Discrete Cosine Transform of an image using SciPy with NumPy fallback."""
    try:
        from scipy.fftpack import dct
        return dct(dct(img.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')
    except Exception:
        return np.abs(np.fft.fft2(img.astype(float)))

def compute_frequency_maps(face_image):
    """
    Computes 2D FFT and 2D DCT log-magnitude spectrum maps.
    Returns a float32 array of shape (2, H, W) containing normalized features.
    """
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_image
        
    # 1. 2D FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    fft_mag = np.log(np.abs(fft_shift) + 1e-8)
    
    # Normalize FFT
    f_min, f_max = fft_mag.min(), fft_mag.max()
    fft_norm = (fft_mag - f_min) / (f_max - f_min + 1e-8) if f_max > f_min else np.zeros_like(fft_mag)
    
    # 2. 2D DCT
    dct_coef = compute_dct2d(gray)
    dct_mag = np.log(np.abs(dct_coef) + 1e-8)
    
    # Normalize DCT
    d_min, d_max = dct_mag.min(), dct_mag.max()
    dct_norm = (dct_mag - d_min) / (d_max - d_min + 1e-8) if d_max > d_min else np.zeros_like(dct_mag)
    
    # Shape: (2, H, W)
    return np.stack([fft_norm, dct_norm], axis=0).astype(np.float32)

def main():
    # Find config file path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, "config.yaml")
    
    # Load config defaults
    cfg = {
        "paths": {
            "raw_dir": "data/raw_videos",
            "preprocessed_dir": "data/preprocessed",
            "metadata_csv": "data/metadata.csv"
        },
        "preprocessing": {
            "frames_per_video": 5,
            "face_size": [224, 224]
        },
        "training": {
            "val_split": 0.15,
            "test_split": 0.15,
            "seed": 42
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_cfg = yaml.safe_load(f)
            if user_cfg:
                for k in cfg:
                    if k in user_cfg:
                        cfg[k].update(user_cfg[k])
                        
    raw_dir = cfg["paths"]["raw_dir"]
    preprocessed_dir = cfg["paths"]["preprocessed_dir"]
    metadata_csv = cfg["paths"]["metadata_csv"]
    
    frames_per_video = cfg["preprocessing"]["frames_per_video"]
    face_size = tuple(cfg["preprocessing"]["face_size"])
    
    val_split = cfg["training"]["val_split"]
    test_split = cfg["training"]["test_split"]
    seed = cfg["training"]["seed"]
    
    # Setup subdirectories
    faces_out_dir = os.path.join(preprocessed_dir, "faces")
    freq_out_dir = os.path.join(preprocessed_dir, "frequency")
    
    os.makedirs(faces_out_dir, exist_ok=True)
    os.makedirs(freq_out_dir, exist_ok=True)
    
    metadata_records = []
    
    # 1. Discover all videos under raw_videos/real and raw_videos/fake
    video_classes = {"real": 0, "fake": 1}
    video_files = []
    
    for class_name, label in video_classes.items():
        class_dir = os.path.join(raw_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist.")
            continue
            
        for f in os.listdir(class_dir):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append({
                    "video_path": os.path.join(class_dir, f),
                    "video_name": f,
                    "label": label,
                    "class_name": class_name
                })
                
    if not video_files:
        print("No raw videos found! Please run generate_synthetic_data.py first.")
        return
        
    print(f"Found {len(video_files)} video files for preprocessing.")
    
    # 2. Extract and process each video
    for item in video_files:
        v_path = item["video_path"]
        v_name = item["video_name"]
        label = item["label"]
        v_base = os.path.splitext(v_name)[0]
        
        print(f"Preprocessing video: {v_name} (Class: {item['class_name']})")
        
        # Extract video frames & crop faces
        frames = extract_video_frames(v_path, num_frames=frames_per_video)
        
        if not frames:
            print(f"Warning: Could not extract frames for {v_name}, skipping.")
            continue
            
        for frame_idx, frame in enumerate(frames):
            face_img = detect_and_crop_face(frame, face_size=face_size)
            
            # Save face crop image
            face_img_name = f"{v_base}_frame_{frame_idx}.jpg"
            face_img_path = os.path.join(faces_out_dir, face_img_name)
            cv2.imwrite(face_img_path, face_img)
            
            # Compute FFT/DCT frequency maps
            freq_maps = compute_frequency_maps(face_img)
            freq_npy_name = f"{v_base}_frame_{frame_idx}.npy"
            freq_npy_path = os.path.join(freq_out_dir, freq_npy_name)
            np.save(freq_npy_path, freq_maps)
            
            # Save pathways in record (store paths relative to workspace root)
            metadata_records.append({
                "video_name": v_name,
                "frame_idx": frame_idx,
                "face_path": os.path.relpath(face_img_path, project_root),
                "freq_path": os.path.relpath(freq_npy_path, project_root),
                "label": label
            })
            
    # 3. Create splits at the VIDEO level to avoid frame-level data leakage
    df = pd.DataFrame(metadata_records)
    if df.empty:
        print("Error: No frames processed successfully.")
        return
        
    unique_videos = df["video_name"].unique()
    
    # Train/Val/Test splits
    train_vids, test_val_vids = train_test_split(
        unique_videos, test_size=(val_split + test_split), random_state=seed
    )
    
    # Split test_val into test and val
    val_ratio_adjusted = val_split / (val_split + test_split)
    val_vids, test_vids = train_test_split(
        test_val_vids, test_size=(1 - val_ratio_adjusted), random_state=seed
    )
    
    video_to_split = {}
    for v in train_vids:
        video_to_split[v] = 'train'
    for v in val_vids:
        video_to_split[v] = 'val'
    for v in test_vids:
        video_to_split[v] = 'test'
        
    df["split"] = df["video_name"].map(video_to_split)
    
    # Save metadata
    os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)
    df.to_csv(metadata_csv, index=False)
    print(f"Preprocessing done! Saved metadata to {metadata_csv} with {len(df)} sample rows.")
    print(f"Splits breakdown: {len(train_vids)} train, {len(val_vids)} val, {len(test_vids)} test videos.")

if __name__ == "__main__":
    main()
