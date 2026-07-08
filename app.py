import os
import sys
import uuid
import yaml
import shutil
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to sys.path to ensure absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.detector import MultiDomainDeepfakeDetector
from src.preprocessing.extract_frames import extract_video_frames, detect_and_crop_face
from src.preprocessing.preprocess import compute_frequency_maps
from src.evaluation.gradcam import GradCAM, find_last_conv_layer, overlay_heatmap

# Load config
config_path = os.path.join(project_root, "config.yaml")
cfg = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

# Path configuration
temp_dir = os.path.join(project_root, "data/temp_inference")
eval_dir = os.path.join(project_root, cfg.get("paths", {}).get("eval_dir", "outputs/evaluation"))
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model globally at startup
model = None
checkpoint_path = os.path.join(
    project_root, 
    cfg.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints"), 
    "best_model.pth"
)

print("Initializing Multi-Domain Deepfake Detector model...")
try:
    model = MultiDomainDeepfakeDetector(cfg=cfg)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model weights successfully from: {checkpoint_path}")
    else:
        print(f"Warning: Model checkpoint not found at {checkpoint_path}. Running with random/initialized weights.")
    
    model = model.to(device)
    model.eval()
except Exception as e:
    print(f"Critical error loading model: {e}")

# Initialize Flask app
app = Flask(__name__, static_folder=eval_dir, static_url_path='/static')
CORS(app)

@app.route("/")
def read_root():
    return jsonify({
        "status": "online",
        "message": "Multi-Domain Deepfake Detection System API is running.",
        "has_model_loaded": model is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return jsonify({"detail": "Detector model could not be initialized. Please check backend logs."}), 500

    # Unique request ID for temp files
    request_id = uuid.uuid4().hex
    temp_files = []
    
    # Defaults/Config
    face_size = tuple(cfg.get("preprocessing", {}).get("face_size", [224, 224]))
    
    face_crop = None
    freq_map = None
    
    try:
        # Check files in request
        video = request.files.get("video")
        image = request.files.get("image")
        
        # Scenario A: Video uploaded
        if video:
            # Save uploaded video
            video_ext = os.path.splitext(video.filename)[1] or ".mp4"
            temp_video_path = os.path.join(temp_dir, f"video_{request_id}{video_ext}")
            video.save(temp_video_path)
            temp_files.append(temp_video_path)
            
            # Extract video frames
            frames = extract_video_frames(temp_video_path, num_frames=3)
            if not frames:
                return jsonify({"detail": "Could not extract frames from the uploaded video. It may be corrupted."}), 400
                
            # Use center frame for face detection & cropping
            middle_frame = frames[len(frames) // 2]
            face_crop = detect_and_crop_face(middle_frame, face_size=face_size)
            
        # Scenario B: Face image uploaded separately
        elif image:
            # Save uploaded image
            img_ext = os.path.splitext(image.filename)[1] or ".jpg"
            temp_img_path = os.path.join(temp_dir, f"image_{request_id}{img_ext}")
            image.save(temp_img_path)
            temp_files.append(temp_img_path)
            
            # Load and crop face
            img_bgr = cv2.imread(temp_img_path)
            if img_bgr is None:
                return jsonify({"detail": "Could not read the uploaded image. Please upload a valid image file."}), 400
            
            face_crop = detect_and_crop_face(img_bgr, face_size=face_size)
            
        else:
            return jsonify({"detail": "You must upload either a video file OR a face image file."}), 400
            
        # Compute frequency maps from face crop
        freq_map = compute_frequency_maps(face_crop)
        
        # Save temp face crop specifically for Grad-CAM overlay read
        temp_face_path = os.path.join(temp_dir, f"face_crop_{request_id}.jpg")
        cv2.imwrite(temp_face_path, face_crop)
        temp_files.append(temp_face_path)
        
        # Transform image to PyTorch tensors
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = transform(face_pil).unsqueeze(0)  # (1, 3, 224, 224)
        
        freq_tensor = torch.tensor(freq_map, dtype=torch.float32).unsqueeze(0)  # (1, 2, 224, 224)
        
        # Move tensors to device
        face_device = face_tensor.to(device)
        freq_device = freq_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            logits, attn_weights = model(face_device, freq_device)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        
        prediction = "FAKE/DEEPFAKE" if fake_prob > real_prob else "REAL"
        confidence = float(max(real_prob, fake_prob) * 100)
        
        # Explainability: Grad-CAM
        heatmap_filename = f"gradcam_{request_id}.png"
        heatmap_path = os.path.join(eval_dir, heatmap_filename)
        heatmap_url = f"/static/{heatmap_filename}"
        
        target_layer = find_last_conv_layer(model.cnn_branch)
        if target_layer is not None:
            try:
                face_device.requires_grad = True
                gradcam = GradCAM(model, target_layer)
                predicted_class = 1 if fake_prob > real_prob else 0
                heatmap = gradcam.generate_heatmap(face_device, freq_device, class_idx=predicted_class)
                gradcam.remove_hooks()
                
                # Overlay heatmap and save
                img_rgb, heatmap_rgb, overlayed = overlay_heatmap(temp_face_path, heatmap)
                
                # Generate a premium dark mode side-by-side visualization
                fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#0b0f19')
                
                axes[0].imshow(img_rgb)
                axes[0].set_title("Analyzed Face Crop", color='white', fontsize=12, fontweight='bold')
                axes[0].axis("off")
                
                axes[1].imshow(overlayed)
                axes[1].set_title(f"Grad-CAM Heatmap (Pred: {prediction})", color='white', fontsize=12, fontweight='bold')
                axes[1].axis("off")
                
                plt.savefig(heatmap_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close()
            except Exception as cam_err:
                print(f"Warning: Grad-CAM failed: {cam_err}")
                cv2.imwrite(heatmap_path, face_crop)
        else:
            cv2.imwrite(heatmap_path, face_crop)
            
        # Clean up temp files
        for fpath in temp_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception:
                pass
                
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "real_prob": round(real_prob * 100, 2),
            "fake_prob": round(fake_prob * 100, 2),
            "heatmap_url": heatmap_url
        })

    except Exception as e:
        # Clean up temp files in case of failure
        for fpath in temp_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception:
                pass
        import traceback
        traceback.print_exc()
        return jsonify({"detail": f"Inference error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
