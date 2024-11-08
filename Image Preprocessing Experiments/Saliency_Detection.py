import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = 'F:/Brain Tumor Detection/Dataset/yes/y20.jpg'
image = cv2.imread(image_path)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2BGR555)

# Create saliency detector object
saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()

# Compute saliency map
success , saliency_map = saliency_detector.computeSaliency(image)

# Noramlize saliency map to the range (0,224)
saliency_map_normalized = cv2.normalize(saliency_map,None,0,224,cv2.NORM_MINMAX)
saliency_map_normalized = np.uint8(saliency_map_normalized)

# Display result
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(saliency_map_normalized,cmap='hot')
plt.title('Saliency Map')
plt.axis('off')

plt.show()