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
from DSU_Net import DSUNet

from PIL import Image
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add train_utils to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_utils'))

# Import evaluation metrics and loss function
try:
    from train_utils.distributed_utils import ConfusionMatrix
    from train_utils.train_and_eval import criterion, create_lr_scheduler
except ImportError as e:
    print(f"Warning: Could not import from train_utils: {e}")
    ConfusionMatrix = None
    criterion = None
    create_lr_scheduler = None

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
        
        # Normalize mask to [0, 1] range if needed
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        return img, mask


base_dir_2017 = "/home/aminu_yusuf/msgunet/datasets/ISIC2017"

# Training transforms with composite data augmentations
# Includes: random flipping, rotation ±15°, brightness/contrast/hue adjustments (±3%)
transform_train = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),                              # Random horizontal flipping
    A.VerticalFlip(p=0.5),                                # Random vertical flipping
    A.Rotate(limit=15, p=0.5),                            # Rotation ±15 degrees
    A.RandomBrightnessContrast(brightness_limit=0.03, contrast_limit=0.03, p=0.5),  # ±3% brightness/contrast
    A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=3, val_shift_limit=3, p=0.5),  # ±3% hue/sat/val
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ToTensorV2()
])

# Test transforms (no augmentation)
transform_test = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Same normalization as training
    ToTensorV2()
])  

train_dataset_2017 = ISICSegmentationDataset(
    base_dir=base_dir_2017, split="train", transform=transform_train, seed=SEED
)
test_dataset_2017 = ISICSegmentationDataset(
    base_dir=base_dir_2017, split="test", transform=transform_test, seed=SEED
)

train_loader_2017 = DataLoader(train_dataset_2017, batch_size=2, shuffle=True, drop_last=True, worker_init_fn=seed_worker)
test_loader_2017 = DataLoader(test_dataset_2017, batch_size=2, shuffle=False, worker_init_fn=seed_worker)

print("Train size:", len(train_dataset_2017))
print("Test size:", len(test_dataset_2017))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize DSU-Net
model = DSUNet(
    n_channels=3, 
    n_classes=1
).to(device)

# AdamW optimizer with paper-specified hyperparameters
optimizer = optim.AdamW(
    model.parameters(), 
    lr=1e-4,
    betas=(0.9, 0.99),
    weight_decay=5e-5
)

# Learning rate scheduler with warmup (official DSU-Net implementation)
num_steps_per_epoch = len(train_loader_2017)
if create_lr_scheduler is not None:
    scheduler = create_lr_scheduler(optimizer, num_steps_per_epoch, epochs=50, warmup=True, warmup_epochs=1, warmup_factor=1e-3)
else:
    # Fallback: simple polynomial decay if import failed
    def lr_lambda(step):
        total_steps = 50 * num_steps_per_epoch
        return (1 - step / total_steps) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(loader, model, criterion, optimizer, scheduler, device, epoch=None, n_epochs=None):
    model.train()
    running_loss = 0.0
    
    # Wrap loader with tqdm
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
    
    for imgs, masks in progress_bar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # DSU-Net returns dict with 'out' and 'outs'
        loss = criterion(outputs, masks, num_classes=1)  # Pass full dict for two-stage loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())  # live update of batch loss
    
    # Update learning rate once per epoch (not per batch)
    scheduler.step()
    
    return running_loss / len(loader)


def eval_epoch(loader, model, criterion, device, epoch=None, n_epochs=None):
    model.eval()
    running_loss = 0.0
    
    # Initialize confusion matrix for comprehensive metrics
    confmat = ConfusionMatrix(num_classes=1) if ConfusionMatrix else None
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False)
        
        for imgs, masks in progress_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)  # DSU-Net returns dict with 'out' and 'outs'
            
            # Calculate loss if criterion is available
            if criterion is not None:
                loss = criterion(outputs, masks, num_classes=1)  # Pass full dict for two-stage loss
            else:
                # Fallback: simple BCE loss if criterion import failed
                loss = torch.nn.BCELoss()(outputs['out'], masks)
            
            running_loss += loss.item()
            
            # Accumulate metrics for confusion matrix
            if confmat is not None:
                preds = (outputs['out'] > 0.5).float()  # Threshold predictions
                confmat.update(masks, preds)
            
            progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(loader)
    
    # Compute and return evaluation metrics
    if confmat is not None:
        acc, sensitivity, specificity, iou, dice = confmat.compute()
        return avg_loss, acc, sensitivity, specificity, iou, dice
    
    return avg_loss, None, None, None, None, None


if __name__ == "__main__":
    # ISIC2017 Training + Testing

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    n_epochs = 50  # DSU-Net paper uses 50 epochs
    train_losses = []
    test_losses = []

    best_loss = float("inf")
    best_model_path = os.path.join(weights_dir, "best_dsunet_isic2017.pth")

    for epoch in range(n_epochs):
        train_loss = train_epoch(train_loader_2017, model, criterion, optimizer, scheduler, device, epoch, n_epochs)
        train_losses.append(train_loss)
        
        eval_results = eval_epoch(test_loader_2017, model, criterion, device, epoch, n_epochs)
        test_loss = eval_results[0]
        test_losses.append(test_loss)
        
        # Display metrics
        if len(eval_results) == 6:
            _, acc, sensitivity, specificity, iou, dice = eval_results
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
            print(f"  Metrics - ACC: {acc*100:.2f}% | SE: {sensitivity*100:.2f}% | SP: {specificity*100:.2f}% | IoU: {iou*100:.2f}% | Dice: {dice*100:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model at epoch {epoch+1} with Test Loss: {test_loss:.4f}")

    # Optionally save final model too
    final_model_path = os.path.join(weights_dir, "dsunet_isic2017.pth")
    torch.save(model.state_dict(), final_model_path)
    print("💾 Training complete, final model saved.")

    model.eval()
    imgs, masks = next(iter(test_loader_2017))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        outputs = model(imgs)  # Returns dict with 'out' and 'outs'
        preds = (outputs['out'] > 0.5).float()  # Threshold predictions to binary

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
        plt.title("DSU-Net Prediction")
        plt.imshow(preds[idx,0].cpu().numpy(), cmap="gray")
        plt.axis('off')

    plt.tight_layout()
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'dsunet_predictions_grid_isic2017.png'))
    plt.show()

    # After training cell (after training loop and model saving)
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DSU-Net - Loss Curve (ISIC2017)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'dsunet_loss_curve_isic2017.png'))


