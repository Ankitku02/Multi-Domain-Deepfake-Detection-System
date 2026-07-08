import os
import sys
import yaml
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.detector import MultiDomainDeepfakeDetector
from src.utils.dataset import MultiDomainDataset

class GradCAM:
    """
    Implements Grad-CAM for visualizing the model's decisions.
    Target layer is dynamically found or specified as the last conv layer of the CNN branch.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, face_tensor, freq_tensor, class_idx=1):
        """
        Generates CAM heatmap for a given input.
        class_idx = 1 represents 'fake'.
        """
        self.model.zero_grad()
        
        # Forward pass (only 2 inputs)
        logits, _ = self.model(face_tensor, freq_tensor)
        
        # Select target class logit
        target_score = logits[0, class_idx]
        
        # Backward pass
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # shape (C, H, W)
        activations = self.activations[0]  # shape (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # shape (C)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU
        cam = torch.clamp(cam, min=0)
        
        # Normalize heatmap to [0, 1]
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def remove_hooks(self):
        """Remove hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

def find_last_conv_layer(module):
    """Recursively searches for the last convolutional layer in a module."""
    last_conv = None
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, nn.Conv2d):
            last_conv = sub_module
    return last_conv

def overlay_heatmap(original_img_path, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlays the generated heatmap onto the original image."""
    img = cv2.imread(original_img_path)
    if img is None:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
    img_resized = cv2.resize(img, (224, 224))
    
    # Scale heatmap to [0, 255]
    heatmap_255 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_255, (224, 224))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(img_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return img_rgb, heatmap_rgb, overlayed

def main():
    # Load configuration
    config_path = os.path.join(project_root, "config.yaml")
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
    metadata_csv = os.path.join(project_root, cfg.get("paths", {}).get("metadata_csv", "data/metadata.csv"))
    checkpoint_path = os.path.join(project_root, cfg.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints"), "best_model.pth")
    eval_dir = os.path.join(project_root, cfg.get("paths", {}).get("eval_dir", "outputs/evaluation"))
    
    # Load test dataset
    if not os.path.exists(metadata_csv):
        print(f"Error: Metadata file not found at {metadata_csv}. Perform preprocessing first.")
        return
        
    dataset = MultiDomainDataset(metadata_csv, split='test', project_root=project_root)
    if len(dataset) == 0:
        print("Warning: Test dataset is empty, falling back to train dataset for Grad-CAM demo.")
        dataset = MultiDomainDataset(metadata_csv, split='train', project_root=project_root)
        
    if len(dataset) == 0:
        print("Error: No data available for Grad-CAM visualization.")
        return
        
    # Pick a sample
    sample_idx = 0
    for idx in range(len(dataset)):
        label = dataset.metadata.iloc[idx]['label']
        if label == 1:
            sample_idx = idx
            break
            
    face_tensor, freq_tensor, label_val = dataset[sample_idx]
    meta_row = dataset.metadata.iloc[sample_idx]
    
    # Prepare batch dimensions (B=1)
    face_batch = face_tensor.unsqueeze(0)
    freq_batch = freq_tensor.unsqueeze(0)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiDomainDeepfakeDetector(cfg=cfg)
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint. Using random weights. Error: {e}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Running Grad-CAM with random weights.")
        
    model = model.to(device)
    model.eval()
    
    # Find target layer in cnn_branch
    target_layer = find_last_conv_layer(model.cnn_branch)
    if target_layer is None:
        print("Error: Could not find any convolutional layer in cnn_branch.")
        return
        
    print(f"Target convolutional layer for Grad-CAM: {target_layer}")
    
    # Instantiate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Move inputs to device
    face_batch = face_batch.to(device)
    freq_batch = freq_batch.to(device)
    
    # Enable gradient tracking on inputs for backward pass
    face_batch.requires_grad = True
    
    # Generate heatmap
    heatmap = gradcam.generate_heatmap(face_batch, freq_batch, class_idx=label_val)
    gradcam.remove_hooks()
    
    # Overlay heatmap onto original image
    face_rel_path = meta_row['face_path']
    face_abs_path = os.path.join(project_root, face_rel_path)
    
    img_rgb, heatmap_rgb, overlayed = overlay_heatmap(face_abs_path, heatmap)
    
    # Save the output
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Face Crop")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_rgb)
    plt.title("Class Activation Heatmap")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlayed)
    plt.title(f"Grad-CAM (Label: {'Fake' if label_val == 1 else 'Real'})")
    plt.axis("off")
    
    output_path = os.path.join(eval_dir, "gradcam_output.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved Grad-CAM visualization to: {output_path}")

if __name__ == "__main__":
    main()
