import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add the project root to path to ensure absolute imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.detector import MultiDomainDeepfakeDetector
from src.utils.dataset import get_dataloader

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (face_imgs, freq_maps, labels) in enumerate(dataloader):
        face_imgs = face_imgs.to(device)
        freq_maps = freq_maps.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (only 2 inputs now)
        logits, _ = model(face_imgs, freq_maps)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item() * face_imgs.size(0)
        _, preds = torch.max(logits, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validates the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for face_imgs, freq_maps, labels in dataloader:
        face_imgs = face_imgs.to(device)
        freq_maps = freq_maps.to(device)
        labels = labels.to(device)
        
        # Forward pass (only 2 inputs now)
        logits, _ = model(face_imgs, freq_maps)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * face_imgs.size(0)
        _, preds = torch.max(logits, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main(epochs_override=None):
    # Load configuration
    config_path = os.path.join(project_root, "config.yaml")
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
    # Resolve paths
    metadata_csv = os.path.join(project_root, cfg.get("paths", {}).get("metadata_csv", "data/metadata.csv"))
    checkpoint_dir = os.path.join(project_root, cfg.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints"))
    log_dir = os.path.join(project_root, cfg.get("paths", {}).get("log_dir", "outputs/logs"))
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Preprocessing and Training Parameters
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    
    batch_size = cfg.get("training", {}).get("batch_size", 4)
    lr = cfg.get("training", {}).get("learning_rate", 0.0001)
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 1e-5))
    epochs = epochs_override if epochs_override is not None else cfg.get("training", {}).get("epochs", 5)
    num_workers = cfg.get("training", {}).get("num_workers", 0)
    
    print("--- Training Settings ---")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Weight Decay: {weight_decay}")
    
    # Initialize DataLoaders
    print("\nLoading datasets...")
    if not os.path.exists(metadata_csv):
        print(f"Error: Metadata file not found at {metadata_csv}. Run preprocessing first.")
        return False
        
    train_loader = get_dataloader(
        metadata_csv, split='train', batch_size=batch_size, shuffle=True,
        num_workers=num_workers, project_root=project_root
    )
    val_loader = get_dataloader(
        metadata_csv, split='val', batch_size=batch_size, shuffle=False,
        num_workers=num_workers, project_root=project_root
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Check if we have samples
    if len(train_loader.dataset) == 0:
        print("Error: No training samples available. Preprocessing split generated empty dataset.")
        return False
        
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model initialization
    model = MultiDomainDeepfakeDetector(cfg=cfg)
    model = model.to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # TensorBoard setup
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging enabled. Logs saved to: {log_dir}")
    except Exception as e:
        print(f"TensorBoard logging unavailable: {e}. Falling back to CSV logging.")
        
    best_val_loss = float('inf')
    history = []
    
    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Log to TensorBoard
        if tb_writer:
            tb_writer.add_scalar("Loss/Train", train_loss, epoch)
            tb_writer.add_scalar("Loss/Val", val_loss, epoch)
            tb_writer.add_scalar("Accuracy/Train", train_acc, epoch)
            tb_writer.add_scalar("Accuracy/Val", val_acc, epoch)
            tb_writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
              
        # Save training history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        # Save checkpoint if best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"--> Saved best model checkpoint to {checkpoint_path}")
            
    # Cleanup Tensorboard
    if tb_writer:
        tb_writer.close()
        
    # Save CSV logs
    history_df = pd.DataFrame(history)
    csv_log_path = os.path.join(log_dir, "training_history.csv")
    history_df.to_csv(csv_log_path, index=False)
    print(f"Training completed successfully! Logs saved to {csv_log_path}")
    return True

if __name__ == "__main__":
    main()
