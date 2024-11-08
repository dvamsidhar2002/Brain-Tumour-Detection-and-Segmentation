import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

# Load a JPG image
def load_jpg_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

# Show the image slice (in this case, the 2D image itself)
def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# Perform skull stripping (or background removal in case of a JPEG image)
def skull_strip(image):
    # Thresholding to create a binary mask
    threshold = np.mean(image) + np.std(image)
    binary_mask = image > threshold
    
    # Morphological operations to refine the mask
    binary_mask = morphology.binary_opening(binary_mask, morphology.disk(2))  # Use disk for 2D images
    binary_mask = morphology.binary_closing(binary_mask, morphology.disk(2))
    
    # Apply the mask to the original image
    stripped_image = image * binary_mask
    return stripped_image, binary_mask

# Load the MRI image in JPG format
file_path = "F:/Brain Tumor Detection/Data/Pituitary/Pituitary_1.jpg"
image_data = load_jpg_image(file_path)

# Show the original image
print("Original MRI image:")
show_image(image_data)

# Perform skull stripping (or background removal)
stripped_image, mask = skull_strip(image_data)

# Show the stripped image
print("Skull stripped MRI image:")
show_image(stripped_image)

# Optionally, save the stripped image as a new JPG file
#cv2.imwrite('skull_stripped_image.jpg', stripped_image)
