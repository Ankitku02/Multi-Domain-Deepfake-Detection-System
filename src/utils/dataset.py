import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MultiDomainDataset(Dataset):
    """
    Custom PyTorch Dataset for loading:
    - Face image (RGB, shape: 3x224x224, with augmentation/normalization)
    - Frequency maps (FFT & DCT, shape: 2x224x224)
    - Classification Label (0: real, 1: fake)
    """
    def __init__(self, metadata_csv, split='train', transform=None, project_root=None):
        super().__init__()
        self.metadata = pd.read_csv(metadata_csv)
        self.split = split
        
        # Filter metadata by split ('train', 'val', or 'test')
        self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        
        if project_root is None:
            # Fallback: assume project root is parent of src directory
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.project_root = project_root
            
        # Default torchvision Transforms
        if transform is not None:
            self.transform = transform
        else:
            # Standard ImageNet normalization for EfficientNet and ViT
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. Load face crop image
        face_rel_path = row['face_path']
        face_abs_path = os.path.join(self.project_root, face_rel_path)
        
        try:
            # Open face crop
            face_img = Image.open(face_abs_path).convert('RGB')
            face_tensor = self.transform(face_img)
        except Exception as e:
            # Fallback if image loading fails
            print(f"Error loading image {face_abs_path}, fallback to zeros. Error: {e}")
            face_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            
        # 2. Load pre-computed frequency map (.npy)
        freq_rel_path = row['freq_path']
        freq_abs_path = os.path.join(self.project_root, freq_rel_path)
        
        try:
            freq_map = np.load(freq_abs_path)
            freq_tensor = torch.tensor(freq_map, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading freq map {freq_abs_path}, fallback to zeros. Error: {e}")
            freq_tensor = torch.zeros((2, 224, 224), dtype=torch.float32)
            
        # 3. Label
        label = int(row['label'])
        
        return face_tensor, freq_tensor, label

def get_dataloader(metadata_csv, split='train', batch_size=4, shuffle=True, num_workers=0, project_root=None):
    """Utility helper to instantiate a DataLoader."""
    dataset = MultiDomainDataset(metadata_csv, split=split, project_root=project_root)
    
    # Shuffle is usually False for val/test
    if split in ['val', 'test']:
        shuffle = False
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and num_workers > 0)
    )
    return loader
