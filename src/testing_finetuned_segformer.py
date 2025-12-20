from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt
import os

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# New number of classes
num_classes = 7

# New image directory
img_path = "../new_dataset/images/output_141.png"  

# Load image
image = Image.open(img_path).convert("RGB")

# Load processor
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

# Load base model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

# Load fine-tuned checkpoint
checkpoint_path = "outputs_new_annotation_1200_300_eph/best_model_fold2.pth"  
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Preprocess image
inputs = feature_extractor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

# Forward pass
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits  # [1, num_classes, H, W]

# Predicted mask
pred_mask = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()

# Updated color map for 7 classes
colors = np.array([
    [0,   0,   0],     # 0 - Background (Black)
    [128, 0,   0],     # 1 - Residential_area (Blue)
    [0,   128, 0],     # 2 - Road (Green)
    [0,   0,   128],   # 3 - River (Red)
    [0,   128, 128],   # 4 - Forest (Yellow = Green + Red)
    [128, 128, 0],     # 5 - Unused_land (Cyan = Blue + Green)
    [128, 0,   128],   # 6 - Agricultural_area (Magenta = Blue + Red)
], dtype=np.uint8)

seg_image = colors[pred_mask]

# Output directory
output_dir = "../new_dataset/predictions"
os.makedirs(output_dir, exist_ok=True)

# Display and save
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg_image)
plt.title("Segmentation")
plt.axis("off")

output_path = os.path.join(output_dir, "segformer_prediction.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Segmentation saved at {output_path}")
