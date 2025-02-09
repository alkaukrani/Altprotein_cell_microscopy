import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
import glob
from skimage.io import imread

# Specify the path to your images (adjust the file extension if needed)
image_files = sorted(glob.glob('/Users/alkadeviukrani/Desktop/APP Cellpose Model/sampleimages/KFE5-day10-2perc-1.jpg'))
images = [imread(fname) for fname in image_files]

# Initialize the Cellpose model (using CPU here, set gpu=True if you have CUDA)
model = models.CellposeModel(
    gpu=False,          # Change to True if using GPU
    model_type='cyto3'  # You can change model_type if needed
)

# Perform segmentation on the list of images
masks, flows, styles = model.eval(
    images,
    channels=[0, 0],         # Use the appropriate channels (here, single channel grayscale)
    diameter=None,           # Let the model estimate cell diameter
    flow_threshold=0.4,      # Adjust this parameter if needed
    cellprob_threshold=0.0,  # Adjust as necessary
    normalize=True,
    invert=False,
    batch_size=4             # Adjust based on your system's memory
)

# Optionally, display the segmentation results for the first image
plt.figure(figsize=(8, 8))
plt.imshow(masks[0], cmap='nipy_spectral')
plt.title('Segmented Masks for First Image')
plt.axis('off')
plt.show()
