import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage

def load_mri_image(file_path):
    # Load the MRI image in grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img

def skull_strip(image):
    # Step 1: Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(image)

    # Step 2: Apply a median filter to smooth out noise while preserving edges
    smooth_img = cv2.medianBlur(enhanced_img, 5)

    # Step 3: Prepare a mask for the floodFill function (2 pixels larger than the input image)
    flood_mask = np.zeros((smooth_img.shape[0] + 2, smooth_img.shape[1] + 2), dtype=np.uint8)

    # Step 4: Use the floodFill function to grow the region from the seed point
    seed_point = (smooth_img.shape[1] // 2, smooth_img.shape[0] // 2)
    cv2.floodFill(smooth_img, flood_mask, seed_point, 255, loDiff=(20,), upDiff=(20,))

    # Step 5: Crop the mask back to the original size and remove small objects from the mask
    flood_mask_cropped = flood_mask[1:-1, 1:-1]
    clean_mask = morphology.remove_small_objects(flood_mask_cropped.astype(bool), min_size=500)

    # Step 6: Fill holes in the cleaned mask to ensure complete brain coverage
    filled_mask = ndimage.binary_fill_holes(clean_mask).astype(np.uint8) * 255

    # Step 7: Apply the mask to the original image to retain only the brain region
    brain_only = cv2.bitwise_and(image, image, mask=filled_mask)

    return brain_only

# Load MRI image
file_path = 'F:/Brain Tumor Detection/Dataset/Training/yes/y3.jpg'  # Replace with your MRI file path
image = load_mri_image(file_path)

# Perform skull stripping
stripped_brain_image = skull_strip(image)

# Plot the original and skull-stripped images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original MRI Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Skull Stripped MRI Image")
plt.imshow(stripped_brain_image, cmap='gray')
plt.axis('off')

plt.show()