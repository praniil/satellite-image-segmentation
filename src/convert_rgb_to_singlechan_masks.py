import numpy as np
from PIL import Image
import os

# BGR → Label ID mapping exactly as given
class_colors = {
    1: (128, 0, 0),    # Residential_area
    2: (0, 128, 0),    # Road
    3: (0, 0, 128),    # River
    4: (0, 128, 128),  # Forest
    5: (128, 128, 0),  # unused_land
    6: (128, 0, 128),  # Agricultural_area
}

mask_train_directory = "../new_dataset/masks"
output_mask_directory = "../new_dataset/label-mask"
os.makedirs(output_mask_directory, exist_ok=True)

for mask_name in os.listdir(mask_train_directory):
    mask_path = os.path.join(mask_train_directory, mask_name)
    mask = np.array(Image.open(mask_path))

    # Initialize label mask as zeros (background = 0)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    # Assign label IDs based on color matches
    for label_id, bgr in class_colors.items():
        matches = np.all(mask == bgr, axis=-1)
        label_mask[matches] = label_id

    # Save single-channel label mask
    output_path = os.path.join(output_mask_directory, mask_name)
    Image.fromarray(label_mask).save(output_path)

print("All masks converted to single-channel labels according  to class_colors mapping!")
