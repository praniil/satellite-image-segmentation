import numpy as np
from PIL import Image
import os

# Your 13-class mapping (BGR → Label ID)
color_to_label = {
    (128, 0, 0): 0,      # Residential_area
    (0, 128, 0): 1,      # Road
    (0, 0, 128): 2,      # River
    (128, 128, 0): 3,    # Recreational_spot
    (128, 0, 128): 4,    # Religious_spot
    (0, 128, 128): 5,    # Forest
    (255, 0, 0): 6,      # Hospital
    (0, 255, 0): 7,      # Airport
    (0, 0, 255): 8,      # Others
    (255, 255, 0): 9,    # unused_land
    (255, 0, 255): 10,   # Agricultural_area
    (0, 255, 255): 11,   # temple
    (255, 128, 0): 12    # Cultural_heritage
}

mask_train_directory = "../dataset/satellite_image_and_mask/Annotation/masks"
output_mask_directory = "../dataset/satellite_image_and_mask/Annotation/label-mask"
os.makedirs(output_mask_directory, exist_ok=True)

for mask_name in os.listdir(mask_train_directory):
    mask_path = os.path.join(mask_train_directory, mask_name)
    mask = np.array(Image.open(mask_path))

    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for color, label in color_to_label.items():
        matches = np.all(mask == color, axis=-1)
        label_mask[matches] = label

    # save the label mask (single-channel)
    output_path = os.path.join(output_mask_directory, mask_name)
    Image.fromarray(label_mask).save(output_path)

print("All masks converted to 13-class single-channel label masks!")
