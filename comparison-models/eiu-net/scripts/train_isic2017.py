import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path to import DSU-Net from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add scripts directory to path for loss_functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import models and loss functions
from network import EIU_Net

from PIL import Image
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Custom implementation of Combined BCE + Dice Loss (0.6*BCE + 0.4*Dice)
class CustomCombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.6, dice_weight=0.4):
        super(CustomCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred_logits, target):
        """Dice loss for logits (applies sigmoid internally)"""
        pred = torch.sigmoid(pred_logits)
        smooth = 1e-5
        
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice_score
    
    def forward(self, pred_logits, target):
        """Combined loss: 0.6*BCE + 0.4*Dice"""
        bce = self.bce_loss(pred_logits, target)
        dice = self.dice_loss(pred_logits, target)
        return self.bce_weight * bce + self.dice_weight * dice


# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class ISICSegmentationDataset(Dataset):
    VALID_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # normal images/masks

    def __init__(self, base_dir, split="train", transform=None, seed=42):
        """
        Args:
            base_dir (str): Path to the ISIC dataset (ISIC2017 or ISIC2018).
            split (str): One of ["train", "test"] for the final 70:30 split.
            transform: Albumentations transform for both images and masks.
            seed (int): Random seed for shuffling.
        """
        self.samples = []
        self.transform = transform

        # Gather images/masks from both train and val folders
        all_samples = []
        for folder in ["train", "val"]:
            img_dir = os.path.join(base_dir, folder, "images")
            mask_dir = os.path.join(base_dir, folder, "masks")
            if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
                continue

            # Keep only normal images (exclude *_superpixels.png)
            images = sorted([
                f for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in self.VALID_IMG_EXTENSIONS
                and "_superpixels" not in f
            ])
            masks = sorted([
                f for f in os.listdir(mask_dir)
                if os.path.splitext(f)[1].lower() in self.VALID_IMG_EXTENSIONS
                and "_superpixels" not in f
            ])

            for i, m in zip(images, masks):
                all_samples.append((os.path.join(img_dir, i), os.path.join(mask_dir, m)))

        # Shuffle once to avoid bias before splitting
        random.Random(seed).shuffle(all_samples)

        # 70:30 split
        split_idx = round(0.7 * len(all_samples))
        if split == "train":
            self.samples = all_samples[:split_idx]
        elif split == "test":
            self.samples = all_samples[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Convert PIL to numpy arrays for albumentations
        img = np.array(img)
        mask = np.array(mask)

        # Apply albumentations transforms (handles image-mask sync automatically)
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        # Add channel dimension to mask for BCEWithLogitsLoss
        # Albumentations ToTensorV2 outputs mask as [H, W], but we need [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        # Convert to float and normalize to [0, 1] range
        mask = mask.float()  # Convert from byte to float
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        return img, mask


base_dir_2017 = "/home/almaan/datasets/ISIC2017"

# Training transforms for EIU-Net
# Input size: 224×320, pixel values scaled to [0, 1]
# As specified in the EIU-Net paper
transform_train = A.Compose([
    A.Resize(224, 320),  # Resize to fixed size (don't crop - may lose lesions)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),  # Matches their degrees=30
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),  # Scale to [0, 1] - just normalize to [0,1]
    ToTensorV2()
])

# Test transforms (no augmentation, just resize)
# Matches their test_type behavior
transform_test = A.Compose([
    A.Resize(224, 320),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),  # Scale to [0, 1]
    ToTensorV2()
])  

train_dataset_2017 = ISICSegmentationDataset(
    base_dir=base_dir_2017, split="train", transform=transform_train, seed=SEED
)
test_dataset_2017 = ISICSegmentationDataset(
    base_dir=base_dir_2017, split="test", transform=transform_test, seed=SEED
)

train_loader_2017 = DataLoader(train_dataset_2017, batch_size=8, shuffle=True, drop_last=True, worker_init_fn=seed_worker)
test_loader_2017 = DataLoader(test_dataset_2017, batch_size=8, shuffle=False, worker_init_fn=seed_worker)

print("Train size:", len(train_dataset_2017))
print("Test size:", len(test_dataset_2017))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize DSU-Net
model = EIU_Net(
    n_channels=3, 
    n_classes=1
).to(device)

# Adam optimizer with EIU-Net specifications
# Initial LR = 0.001, Weight decay = 0.0001
optimizer = optim.Adam(
    model.parameters(), 
    lr=1e-3,
    weight_decay=1e-4
)

# CosineAnnealingWarmRestarts scheduler (150 epochs total)
# Matches their code: T_0=10, T_mult=2
# Cycle 1: epochs 0-10, Cycle 2: 10-30, Cycle 3: 30-70, Cycle 4: 70-150
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,   # First cycle is 10 epochs
    T_mult=2,  # Each subsequent cycle doubles in length
    eta_min=1e-6  # Minimum learning rate
)

# Initialize loss function (0.6*BCE + 0.4*Dice with logits) - custom implementation
criterion = CustomCombinedLoss(bce_weight=0.6, dice_weight=0.4)


def train_epoch(loader, model, criterion, optimizer, scheduler, device, epoch=None, n_epochs=None):
    model.train()
    running_loss = 0.0
    
    # Wrap loader with tqdm
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
    
    for imgs, masks in progress_bar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # EIU-Net model outputs logits (no sigmoid)
        
        # Calculate BCE + Dice loss (loss function applies sigmoid internally)
        loss = criterion(outputs, masks)
        if loss.isnan():
             print(f"NaN detected\nBatch loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    # Update learning rate once per epoch (not per batch)
    scheduler.step()
    
    return running_loss / len(loader)


def eval_epoch(loader, model, criterion, device, epoch=None, n_epochs=None):
    model.eval()
    running_loss = 0.0
    nan_count = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False)
        
        for batch_idx, (imgs, masks) in enumerate(progress_bar):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)  # EIU-Net model outputs logits
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                print(f"\n⚠️  NaN in network outputs at batch {batch_idx}, replacing with zeros")
                outputs = torch.nan_to_num(outputs, nan=0.0)  # Replace NaN with 0
                nan_count += 1
            
            # Calculate BCE + Dice loss (loss function applies sigmoid internally)
            loss = criterion(outputs, masks)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"\n⚠️  NaN in loss at batch {batch_idx}")
                print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
                print(f"Mask range: [{masks.min():.4f}, {masks.max():.4f}]")
                nan_count += 1
                continue
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    
    if nan_count > 0:
        print(f"\n⚠️  Total NaN batches: {nan_count}/{len(loader)}")
    
    avg_loss = running_loss / max(1, len(loader) - nan_count)
    return avg_loss


if __name__ == "__main__":
    # EIU-Net ISIC2017 Training + Testing

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    n_epochs = 150
    train_losses = []
    test_losses = []

    best_loss = float("inf")
    best_model_path = os.path.join(weights_dir, "best_eiuenet_isic2017.pth")

    for epoch in range(n_epochs):
        train_loss = train_epoch(train_loader_2017, model, criterion, optimizer, scheduler, device, epoch, n_epochs)
        train_losses.append(train_loss)
        
        # Make sure model is set to eval mode before evaluating
        model.eval()
        test_loss = eval_epoch(test_loader_2017, model, criterion, device, epoch, n_epochs)
        test_losses.append(test_loss)
        
        # Switch back to train mode for next epoch
        model.train()
        
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model at epoch {epoch+1} with Test Loss: {test_loss:.4f}")

    # Optionally save final model too
    final_model_path = os.path.join(weights_dir, "eiuenet_isic2017.pth")
    torch.save(model.state_dict(), final_model_path)
    print("💾 Training complete, final model saved.")

    model.eval()
    imgs, masks = next(iter(test_loader_2017))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        outputs = model(imgs)  # EIU-Net returns logits
        preds = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid then threshold

    n_samples = min(6, imgs.shape[0])
    plt.figure(figsize=(12, n_samples * 3))
    for idx in range(n_samples):
        # Base image
        plt.subplot(n_samples, 3, idx * 3 + 1)
        plt.title(f"Image {idx+1}")
        plt.imshow(np.transpose(imgs[idx].cpu().numpy(), (1,2,0)))
        plt.axis('off')
        # Ground truth mask
        plt.subplot(n_samples, 3, idx * 3 + 2)
        plt.title("Mask")
        plt.imshow(masks[idx,0].cpu().numpy(), cmap="gray")
        plt.axis('off')
        # Prediction
        plt.subplot(n_samples, 3, idx * 3 + 3)
        plt.title("EIU-Net Prediction")
        plt.imshow(preds[idx,0].cpu().numpy(), cmap="gray")
        plt.axis('off')

    plt.tight_layout()
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'eiuenet_predictions_grid_isic2017.png'))
    plt.show()

    # After training cell (after training loop and model saving)
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EIU-Net - Loss Curve (ISIC2017)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'eiuenet_loss_curve_isic2017.png'))


