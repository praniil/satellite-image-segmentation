import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F
from sklearn.model_selection import KFold
import wandb

# ==============================
# CONFIGURATION
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 13
batch_size = 16          # increased batch size
learning_rate = 3e-5
num_epochs = 100
image_size = 256
patience = 7
num_folds = 2            # fewer folds for small dataset

train_img_dir = "../dataset/satellite_image_and_mask/Annotation/images"
train_mask_dir = "../dataset/satellite_image_and_mask/Annotation/label-mask"

output_dir = "./segformer_ktm_datasets"
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = os.path.join(output_dir, "kfold_training_log.csv")

# ==============================
# DATASET
# ==============================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, feature_extractor, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.feature_extractor = feature_extractor
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, self.masks[idx])))

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        encoded = feature_extractor(images=img, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze()

        # Fix: convert mask safely
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).long()
        else:
            mask_tensor = mask.long()  # already a tensor

        return pixel_values, mask_tensor


# ==============================
# DATA AUGMENTATIONS
# ==============================
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Resize(image_size, image_size),
    ToTensorV2()
])

# ==============================
# FEATURE EXTRACTOR
# ==============================
feature_extractor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"  # smaller model
)

# ==============================
# METRICS
# ==============================
def mean_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    target_resized = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[1:], mode="nearest").squeeze(1).long()
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target_resized == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            continue
        ious.append(intersection / union)
    return torch.mean(torch.tensor(ious)) if ious else torch.tensor(0.0)

# ==============================
# DATASET AND K-FOLD
# ==============================
dataset = SegDataset(train_img_dir, train_mask_dir, feature_extractor, train_tf)
all_indices = np.arange(len(dataset))
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# CSV logging
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "epoch", "train_loss", "train_iou", "val_loss", "val_iou"])

fold_metrics = []

# ==============================
# K-FOLD TRAINING
# ==============================
for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices), 1):
    print(f"\n========== Fold {fold}/{num_folds} ==========")
    torch.cuda.empty_cache()

    # Dataloaders
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ==============================
    # MODEL
    # ==============================
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    # No freezing

    # ==============================
    # CLASS WEIGHTS
    # ==============================
    class_pixel_counts = np.zeros(num_classes)
    for idx in train_idx:
        mask = np.array(Image.open(os.path.join(train_mask_dir, dataset.masks[idx])))
        for cls in range(num_classes):
            class_pixel_counts[cls] += (mask == cls).sum()
    class_weights = 1.0 / (class_pixel_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device='cuda')

    wandb.init(project="segformer_ktm_datasets", name=f"fold_{fold}", reinit=True)
    wandb.config.update({
        "fold": fold,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "image_size": image_size,
        "patience": patience
    })

    best_val_iou = 0.0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        # TRAINING
        model.train()
        total_train_loss = 0.0
        total_train_iou = 0.0

        for step, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Training Fold {fold} Epoch {epoch+1}")):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=imgs, labels=masks)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            total_train_iou += mean_iou(outputs.logits.detach(), masks, num_classes).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_iou = total_train_iou / len(train_loader)

        # VALIDATION
        model.eval()
        total_val_loss = 0.0
        total_val_iou = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Validation Fold {fold}"):
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=imgs, labels=masks)
                    loss = outputs.loss

                total_val_loss += loss.item()
                total_val_iou += mean_iou(outputs.logits, masks, num_classes).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)

        # LOGGING
        print(f"\nFold [{fold}] Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val IoU:   {avg_val_iou:.4f}")

        wandb.log({
            "Train Loss": avg_train_loss,
            "Train IoU": avg_train_iou,
            "Val Loss": avg_val_loss,
            "Val IoU": avg_val_iou,
            "Epoch": epoch + 1,
            "Fold": fold
        })

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([fold, epoch+1, avg_train_loss, avg_train_iou, avg_val_loss, avg_val_iou])

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"fold{fold}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # Early stopping
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            no_improve_epochs = 0
            best_model_path = os.path.join(checkpoint_dir, f"best_model_fold{fold}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved for fold {fold} at epoch {epoch+1} with Val IoU = {best_val_iou:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered for fold {fold} at epoch {epoch+1}.")
            break

    fold_metrics.append(best_val_iou)
    wandb.finish()
    torch.cuda.empty_cache()

# ==============================
# SUMMARY
# ==============================
print("\n========== K-FOLD CROSS VALIDATION RESULTS ==========")
for i, iou in enumerate(fold_metrics, 1):
    print(f"Fold {i} Best Val IoU: {iou:.4f}")
print(f"Mean Best Val IoU Across Folds: {np.mean(fold_metrics):.4f}")
