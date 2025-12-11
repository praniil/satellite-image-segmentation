import os

images = os.listdir("../dataset/satellite_image_and_mask/Annotation/images")
masks = os.listdir("../dataset/satellite_image_and_mask/Annotation/masks")
label_masks = os.listdir("../dataset/satellite_image_and_mask/Annotation/label-mask")

print(len(images))
print(len(masks))
print(len(label_masks))