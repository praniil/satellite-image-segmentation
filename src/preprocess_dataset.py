import os

images = os.listdir("../new_dataset/images")
masks = os.listdir("../new_dataset/masks")
label_masks = os.listdir("../new_dataset/label-mask")

print(len(images))
print(len(masks))
print(len(label_masks))

extra = set(images) - set(masks)
print(extra)