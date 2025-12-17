import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.model_selection import KFold
import wandb

# ================= CONFIG =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 7  # background (0) + 6 label classes
batch_size = 16
learning_rate = 3e-5
num_epochs = 100
image_size = 512  # increased from 256 for better SegFormer performance
patience = 10
num_folds = 5

img_dir = "../new_dataset/images"
mask_dir = "../new_dataset/label-mask"

output_dir = "./outputs_new_annotation_1200"
os.makedirs(output_dir, exist_ok=True)

# ================= DATASET =================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor, transforms):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.processor = processor
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, self.masks[idx])))

        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"].long()

        # Processor applied AFTER augmentations
        encoded = self.processor(images=img, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)

        return pixel_values, mask

# ================= AUGMENTATION =================
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Resize(image_size, image_size),
    ToTensorV2()
])

# ================= PROCESSOR =================
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

# ================= METRICS =================
def mean_iou(logits, targets):
    preds = logits.argmax(dim=1)
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (targets == c)).sum().float()
        union = ((preds == c) | (targets == c)).sum().float()
        if union > 0:
            ious.append(inter / union)
    if len(ious) == 0:
        return torch.tensor(0.0)
    return torch.mean(torch.stack(ious)).item()

# ================= DATASET =================
dataset = SegDataset(img_dir, mask_dir, processor, train_tf)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# ================= COMPUTE CLASS WEIGHTS =================
print("Computing class weights...")
class_counts = np.zeros(num_classes)
for _, mask in dataset:
    mask_np = mask.numpy()
    for c in range(num_classes):
        class_counts[c] += (mask_np == c).sum()

class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)

# ================= TRAINING =================
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):

    print(f"\n========== FOLD {fold}/{num_folds} ==========")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    # Optional: use focal loss
    # from torchmetrics.classification import MulticlassFocalLoss
    # criterion = MulticlassFocalLoss(num_classes=num_classes, alpha=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=True)

    wandb.init(
        project="segformer-new-annotation_1200",
        name=f"fold-{fold}",
        config={
            "fold": fold,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "image_size": image_size
        }
    )

    best_miou = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(pixel_values=imgs)
                logits = F.interpolate(
                    outputs.logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        val_miou = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                with autocast(device_type="cuda"):
                    outputs = model(pixel_values=imgs)
                    logits = F.interpolate(
                        outputs.logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    loss = criterion(logits, masks)

                val_loss += loss.item()
                val_miou += mean_iou(logits, masks)

        val_loss /= len(val_loader)
        val_miou /= len(val_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou
        })

        print(
            f"Epoch {epoch+1} | "
            f"TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"mIoU {val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"best_model_fold{fold}.pth")
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    fold_results.append(best_miou)
    wandb.finish()
    torch.cuda.empty_cache()

# ================= FINAL K-FOLD RESULT =================
mean_score = np.mean(fold_results)
std_score = np.std(fold_results)

print("\n========== FINAL RESULT ==========")
print(f"SegFormer-B0 : {mean_score:.4f} ± {std_score:.4f}")
