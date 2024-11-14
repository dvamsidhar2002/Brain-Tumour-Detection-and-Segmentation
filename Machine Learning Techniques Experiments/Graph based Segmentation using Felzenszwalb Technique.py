import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
import numpy as np

# Load and preprocess the MRI image
def load_and_preprocess_image(image_path):
    # Load the image as grayscale and normalize to [0, 1] range
    image = io.imread(image_path, as_gray=True)
    image = img_as_float(image)  # Convert to floating-point for better precision
    return image

# Apply Felzenszwalb's Graph-Based Segmentation
def apply_felzenszwalb_segmentation(image, scale=100, sigma=0.5, min_size=50):
    # Use Felzenszwalb's method for image segmentation
    segmented_image = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
    return segmented_image

# Isolate tumor region by finding the largest or brightest segment
def isolate_tumor_segment(segmented_image, original_image):
    # Find unique segments and their sizes
    unique_segments, counts = np.unique(segmented_image, return_counts=True)
    
    # Assuming the tumor is in the segment with the highest average intensity in the original image
    tumor_segment = None
    max_intensity = 0
    for segment in unique_segments:
        mask = (segmented_image == segment)
        segment_intensity = np.mean(original_image[mask])
        
        if segment_intensity > max_intensity:
            max_intensity = segment_intensity
            tumor_segment = segment

    # Create a binary mask for the tumor region
    tumor_mask = (segmented_image == tumor_segment)
    
    return tumor_mask

# Function to display the segmentation results with highlighted tumor
def display_segmentation_results(original_image, tumor_mask):
    # Display original and segmented images side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original MRI Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(original_image, cmap='gray')
    plt.imshow(tumor_mask, cmap='jet', alpha=0.5)  # Overlay tumor mask
    plt.title("Highlighted Tumor Region")
    plt.axis('off')

    plt.show()

# Load the MRI image (replace 'path_to_image.jpg' with your actual image path)
image_path = 'F:/Brain Tumor Detection/Dataset/Training/yes/y0.jpg'  # Example path
mri_image = load_and_preprocess_image(image_path)

# Perform Felzenszwalb segmentation
segmented_image = apply_felzenszwalb_segmentation(mri_image, scale=100, sigma=0.5, min_size=50)

# Isolate the tumor segment
tumor_mask = isolate_tumor_segment(segmented_image, mri_image)

# Display results with the tumor highlighted
display_segmentation_results(mri_image, tumor_mask)
