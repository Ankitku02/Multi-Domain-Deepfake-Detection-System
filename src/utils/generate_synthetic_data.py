import os
import cv2
import numpy as np

def generate_video(file_path, is_fake=False, width=320, height=240, fps=10, duration=3.0):
    """Generates a synthetic MP4 video with a moving circle representing a face/object."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    num_frames = int(duration * fps)
    
    # Use MP4V codec which is widely compatible on Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    
    # Draw simple shapes to simulate visual features
    for frame_idx in range(num_frames):
        # Create a black image
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw moving background lines (different for real and fake to simulate patterns)
        for i in range(0, width, 40):
            offset = int(frame_idx * (2 if is_fake else 1)) % 40
            cv2.line(frame, (i + offset, 0), (i + offset, height), (30, 30, 30), 2)
            
        # Draw a moving oval structure to simulate a face/head
        # Center coordinates move with time
        cx = int(width / 2 + 30 * np.sin(2 * np.pi * frame_idx / num_frames))
        cy = int(height / 2 + 20 * np.cos(2 * np.pi * frame_idx / num_frames))
        
        # Face base color (skin-like or distinct for real/fake classification)
        face_color = (150, 180, 230) if not is_fake else (130, 150, 210)
        cv2.ellipse(frame, (cx, cy), (50, 70), 0, 0, 360, face_color, -1)
        
        # Eyes
        cv2.circle(frame, (cx - 15, cy - 20), 8, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 15, cy - 20), 8, (255, 255, 255), -1)
        # Pupils
        cv2.circle(frame, (cx - 15, cy - 20), 3, (0, 0, 0), -1)
        cv2.circle(frame, (cx + 15, cy - 20), 3, (0, 0, 0), -1)
        
        # Mouth
        cv2.ellipse(frame, (cx, cy + 25), (15, 10), 0, 0, 180, (0, 0, 255), -1)
        
        # Add some compression noise or frequency pattern if it is fake
        if is_fake:
            noise = np.random.normal(0, 15, (height, width, 3)).astype(np.uint8)
            frame = cv2.add(frame, noise)
            
        out.write(frame)
        
    out.release()
    print(f"Generated video: {file_path}")

def main():
    import yaml
    # Load config to see output paths if config exists
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    
    raw_dir = "data/raw_videos"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                raw_dir = cfg.get("paths", {}).get("raw_dir", raw_dir)
        except Exception:
            pass
            
    real_dir = os.path.join(raw_dir, "real")
    fake_dir = os.path.join(raw_dir, "fake")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Generate 2 real videos
    generate_video(os.path.join(real_dir, "real_1.mp4"), is_fake=False)
    generate_video(os.path.join(real_dir, "real_2.mp4"), is_fake=False)
    
    # Generate 2 fake videos
    generate_video(os.path.join(fake_dir, "fake_1.mp4"), is_fake=True)
    generate_video(os.path.join(fake_dir, "fake_2.mp4"), is_fake=True)
    
    print("Synthetic data generation completed successfully!")

if __name__ == "__main__":
    main()
