import os
import sys
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path to ensure absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.detector import MultiDomainDeepfakeDetector
from src.preprocessing.extract_frames import extract_video_frames, detect_and_crop_face
from src.preprocessing.preprocess import compute_frequency_maps

def preprocess_and_predict(video_path=None, image_path=None, checkpoint_path=None):
    """
    Main pipeline for prediction.
    Accepts either:
    1. A raw video path
    2. A face image path
    """
    # 1. Load config
    config_path = os.path.join(project_root, "config.yaml")
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
    # Load defaults
    face_size = tuple(cfg.get("preprocessing", {}).get("face_size", [224, 224]))
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(project_root, cfg.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints"), "best_model.pth")
        
    temp_dir = os.path.join(project_root, "data/temp_inference")
    os.makedirs(temp_dir, exist_ok=True)
    
    face_crop = None
    freq_map = None
    
    # 2. Extract inputs depending on arguments
    if video_path is not None:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
            
        print(f"\nProcessing video: {video_path}...")
        
        # Extract frame & face
        frames = extract_video_frames(video_path, num_frames=3)
        if not frames:
            print("Error: Could not extract frames from video.")
            return
            
        # Use center frame for inference
        middle_frame = frames[len(frames) // 2]
        face_crop = detect_and_crop_face(middle_frame, face_size=face_size)
        
        # Save temp crop for Grad-CAM overlay
        temp_face_path = os.path.join(temp_dir, "face_temp.jpg")
        cv2.imwrite(temp_face_path, face_crop)
        
        # Compute frequency map
        freq_map = compute_frequency_maps(face_crop)
        
    elif image_path is not None:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return
            
        print(f"\nProcessing separate Image: {image_path}...")
        
        # Load image and detect face
        img_bgr = cv2.imread(image_path)
        face_crop = detect_and_crop_face(img_bgr, face_size=face_size)
        temp_face_path = os.path.join(temp_dir, "face_temp.jpg")
        cv2.imwrite(temp_face_path, face_crop)
        
        # Compute frequency maps
        freq_map = compute_frequency_maps(face_crop)
    else:
        print("Error: You must provide either --video OR --image.")
        return
        
    # 3. Apply standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Convert inputs to PyTorch tensors and add batch dimensions
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_tensor = transform(face_pil).unsqueeze(0)  # (1, 3, 224, 224)
    
    freq_tensor = torch.tensor(freq_map, dtype=torch.float32).unsqueeze(0)  # (1, 2, 224, 224)
    
    # 4. Load detector model and weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiDomainDeepfakeDetector(cfg=cfg)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Train the model first.")
        return
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model weights from: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model = model.to(device)
    model.eval()
    
    # 5. Run inference
    with torch.no_grad():
        face_device = face_tensor.to(device)
        freq_device = freq_tensor.to(device)
        
        logits, attn_weights = model(face_device, freq_device)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
    real_prob = probs[0]
    fake_prob = probs[1]
    
    prediction = "FAKE/DEEPFAKE" if fake_prob > real_prob else "REAL"
    confidence = max(real_prob, fake_prob) * 100
    
    print("\n" + "="*40)
    print(f"      INFERENCE PREDICTION RESULT")
    print("="*40)
    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence:.2f}%")
    print(f"Details    : Real={real_prob*100:.1f}%, Fake={fake_prob*100:.1f}%")
    print("="*40)
    
    # 6. Generate explainability heatmap (Grad-CAM)
    print("\nGenerating Grad-CAM visualization map...")
    from src.evaluation.gradcam import GradCAM, find_last_conv_layer, overlay_heatmap
    target_layer = find_last_conv_layer(model.cnn_branch)
    if target_layer is not None:
        try:
            # Re-enable gradient tracking
            face_device.requires_grad = True
            
            gradcam = GradCAM(model, target_layer)
            # Generate heatmap for the predicted class
            predicted_class = 1 if fake_prob > real_prob else 0
            heatmap = gradcam.generate_heatmap(face_device, freq_device, class_idx=predicted_class)
            gradcam.remove_hooks()
            
            # Save visual output overlay
            temp_face_path = os.path.join(temp_dir, "face_temp.jpg")
            img_rgb, heatmap_rgb, overlayed = overlay_heatmap(temp_face_path, heatmap)
            
            output_dir = os.path.join(project_root, cfg.get("paths", {}).get("eval_dir", "outputs/evaluation"))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "prediction_gradcam.png")
            
            # Save image
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title("Analyzed Face Crop")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(overlayed)
            plt.title(f"Grad-CAM Highlight (Pred: {prediction})")
            plt.axis("off")
            
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved Grad-CAM heatmap to: {output_path}")
        except Exception as e:
            print(f"Warning: Grad-CAM generation failed. Error: {e}")
            
    print("Prediction run successfully completed.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict single sample Deepfake status")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--image", type=str, default=None, help="Path to face image file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    preprocess_and_predict(
        video_path=args.video,
        image_path=args.image,
        checkpoint_path=args.checkpoint
    )
