from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 13

# Load the image
img_path = "../dataset/satellite_image_and_mask/Annotation/images/output_661.png"
image = Image.open(img_path).convert("RGB")

# Load processor
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Load base model and then load your fine-tuned checkpoint
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

# Load your fine-tuned weights
# src/finetuning_segformer/segformer_after_regularization
# src/finetuning_segformer/segformer_improved_unfreezed_ encoder/checkpoints/best_model_fold1.pth
checkpoint_path = "segformer_ktm_datasets/checkpoints/best_model_fold1.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Preprocess image
inputs = feature_extractor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

# Forward pass
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits  # shape: [1, num_classes, H, W]

# Get predicted class per pixel
pred_mask = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()

# Color map

colors = np.array([
    [128,   0,   0],   # 0 - Residential_area (mostly blue → dark blue)
    [  0, 128,   0],   # 1 - Road (mostly green → dark green)
    [  0,   0, 128],   # 2 - River (mostly red → dark red)
    [128, 128,   0],   # 3 - Recreational_spot (blue+green → olive)
    [128,   0, 128],   # 4 - Religious_spot (blue+red → purple)
    [  0, 128, 128],   # 5 - Forest (green+red → yellowish / brownish)
    [255,   0,   0],   # 6 - Hospital (mostly blue → bright blue)
    [  0, 255,   0],   # 7 - Airport (mostly green → bright green)
    [  0,   0, 255],   # 8 - Others (mostly red → bright red)
    [255, 255,   0],   # 9 - Unused_land (blue+green → cyan)
    [255,   0, 255],   # 10 - Agricultural_area (blue+red → magenta)
    [  0, 255, 255],   # 11 - Temple (green+red → yellow)
    [255, 128,   0],   # 12 - Cultural_heritage (blue+green+red → orange)
], dtype=np.uint8)

seg_image = colors[pred_mask]

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg_image)
plt.title("Segmentation")
plt.axis("off")
plt.savefig("segformer_prediction.png", dpi=300, bbox_inches="tight")
plt.close()
