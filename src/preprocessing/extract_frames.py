import os
import cv2
import numpy as np

# Global face detector variable, initialized lazily to save import time and avoid failures
_mtcnn_detector = None

def get_face_detector():
    """Lazily initializes and returns the MTCNN detector, fallback to None if import/init fails."""
    global _mtcnn_detector
    if _mtcnn_detector is None:
        try:
            from mtcnn import MTCNN
            # Suppress excessive logging if possible
            _mtcnn_detector = MTCNN()
            print("MTCNN face detector initialized successfully.")
        except Exception as e:
            print(f"Warning: MTCNN could not be initialized. Center crop fallback will be used. Error: {e}")
            _mtcnn_detector = False
    return _mtcnn_detector if _mtcnn_detector is not False else None

def extract_video_frames(video_path, num_frames=5):
    """Extracts a specified number of frames evenly distributed across the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
        
    # Get evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Fallback to reading sequentially if set fails
            ret, frame = cap.read()
            if not ret:
                break
        frames.append(frame)
        
    cap.release()
    return frames

def detect_and_crop_face(frame, face_size=(224, 224)):
    """
    Detects the main face in the frame using MTCNN.
    Falls back to cropping the center of the frame if no face is found or MTCNN is unavailable.
    """
    h, w, _ = frame.shape
    detector = get_face_detector()
    
    if detector is not None:
        try:
            # MTCNN expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb_frame)
            if results:
                # Get the face with the highest confidence
                best_face = max(results, key=lambda x: x['box'][4] if len(x['box']) > 4 else x['confidence'])
                x, y, width, height = best_face['box']
                
                # Expand box slightly to include more face context
                padding_w = int(width * 0.1)
                padding_h = int(height * 0.1)
                
                x1 = max(0, x - padding_w)
                y1 = max(0, y - padding_h)
                x2 = min(w, x + width + padding_w)
                y2 = min(h, y + height + padding_h)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    return cv2.resize(face_crop, face_size)
        except Exception as e:
            # Silent fallback to center crop on MTCNN processing error
            pass
            
    # Center Crop Fallback
    crop_w = min(w, face_size[0] * 2)
    crop_h = min(h, face_size[1] * 2)
    cx, cy = w // 2, h // 2
    
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, cx + crop_w // 2)
    y2 = min(h, cy + crop_h // 2)
    
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size > 0:
        return cv2.resize(face_crop, face_size)
    else:
        # Emergency return of scaled original frame
        return cv2.resize(frame, face_size)
