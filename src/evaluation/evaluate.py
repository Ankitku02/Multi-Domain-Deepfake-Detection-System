import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay

# Add the project root to path to ensure absolute imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.detector import MultiDomainDeepfakeDetector
from src.utils.dataset import get_dataloader

def calculate_eer(labels, y_probs):
    """Calculates the Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(labels, y_probs, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Runs evaluation and returns true labels, predictions, and probabilities."""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []  # Probability of class 1 (fake)
    
    for face_imgs, freq_maps, labels in dataloader:
        face_imgs = face_imgs.to(device)
        freq_maps = freq_maps.to(device)
        
        logits, _ = model(face_imgs, freq_maps)
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)
        
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Index 1 is 'fake'
        
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Multi-Domain Deepfake Detector")
    parser.add_argument("--metadata", type=str, default=None, help="Path to alternative metadata CSV for cross-dataset testing")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model checkpoint")
    args = parser.parse_args(args=[] if 'ipykernel' in sys.modules else None)
    
    # Load configuration
    config_path = os.path.join(project_root, "config.yaml")
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
    # Resolve paths
    metadata_csv = args.metadata if args.metadata else os.path.join(project_root, cfg.get("paths", {}).get("metadata_csv", "data/metadata.csv"))
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(project_root, cfg.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints"), "best_model.pth")
    eval_dir = os.path.join(project_root, cfg.get("paths", {}).get("eval_dir", "outputs/evaluation"))
    
    os.makedirs(eval_dir, exist_ok=True)
    
    print("--- Evaluation Settings ---")
    print(f"Metadata file: {metadata_csv}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluation outputs: {eval_dir}\n")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    if not os.path.exists(metadata_csv):
        print(f"Error: Metadata file not found at {metadata_csv}. Perform preprocessing first.")
        return
        
    df_meta = pd.read_csv(metadata_csv)
    if 'split' in df_meta.columns and args.metadata is None:
        print("Evaluating model on 'test' split...")
        test_loader = get_dataloader(
            metadata_csv, split='test', batch_size=cfg.get("training", {}).get("batch_size", 4),
            shuffle=False, num_workers=0, project_root=project_root
        )
    else:
        print("Cross-dataset mode: Evaluating model on ALL records inside the metadata file...")
        from src.utils.dataset import MultiDomainDataset
        dataset = MultiDomainDataset(metadata_csv, split='train', project_root=project_root)
        dataset.metadata = df_meta
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        
    print(f"Test split sample count: {len(test_loader.dataset)}")
    if len(test_loader.dataset) == 0:
        print("Warning: Test dataset contains 0 samples. Skipping evaluation.")
        return
        
    # Load model architecture
    model = MultiDomainDeepfakeDetector(cfg=cfg)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Train the model first.")
        return
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model weights from epoch {checkpoint.get('epoch', 'N/A')}.")
    except Exception as e:
        print(f"Error loading checkpoint weights: {e}")
        return
        
    model = model.to(device)
    
    # Run evaluation
    labels, preds, probs = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        eer, eer_threshold = calculate_eer(labels, probs)
    else:
        auc = 0.0
        eer = 0.0
        eer_threshold = 0.5
        print("Warning: Only a single class was present in the test labels. AUC and EER cannot be computed correctly.")
        
    metrics = {
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "f1_score": float(f1),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "total_samples": int(len(labels))
    }
    
    # Display results
    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"EER      : {eer*100:.2f}% (Threshold: {eer_threshold:.4f})")
    print(f"Samples  : {len(labels)}")
    
    # Save metrics JSON
    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics summary to: {metrics_path}")
    
    # 1. Plot and save Confusion Matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix plot to: {cm_path}")
    
    # 2. Plot and save ROC Curve
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        roc_path = os.path.join(eval_dir, "roc_curve.png")
        plt.savefig(roc_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved ROC Curve plot to: {roc_path}")
        
    print("\nEvaluation successfully completed.")
 
if __name__ == "__main__":
    main()
