'''skimage.io — for loading images

skimage.color — for color space conversions (e.g., RGB to grayscale)

skimage.feature — for texture features like GLCM and Haralick features 

     GLCM(Gray Level Co-occurrence Matrix) is a matrix that basically comapres the pixel intensity values of an image with its neighboring pixels 
     and then decides on features like contrast, correlation, energy, and homogeneity. 

    Haralick features a set of 13 features that are derived from GLCM and give a good representation of texture in an image.

skimage.measure — for shape analysis and region properties (area, perimeter, eccentricity, etc.)

'''

import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
import mahotas

# Load image (grayscale)
image = cv2.imread("your_image.png", cv2.IMREAD_GRAYSCALE)

# Make sure it's 8-bit
if image.max() > 255:
    image = cv2.convertScaleAbs(image)


# GLCM Features (scikit-image)


# Define distances and angles
distances = [1, 2, 4]  # pixel pair distances
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles in radians

# Compute GLCM
glcm = greycomatrix(image, 
                    distances=distances, 
                    angles=angles, 
                    symmetric=True, 
                    normed=True)

# Extract some properties
contrast = greycoprops(glcm, 'contrast')
energy = greycoprops(glcm, 'energy')
homogeneity = greycoprops(glcm, 'homogeneity')
correlation = greycoprops(glcm, 'correlation')

print("GLCM Contrast:\n", contrast)
print("GLCM Energy:\n", energy)
print("GLCM Homogeneity:\n", homogeneity)
print("GLCM Correlation:\n", correlation)


# Haralick Features (mahotas)

haralick_features = mahotas.features.haralick(image).mean(axis=0)
print("\nHaralick Features:\n", haralick_features)

'''
. Angular Second Moment (ASM) / Energy
What it measures: Texture uniformity (how consistent the pixel pairs are).

High value: Image is uniform (e.g., all pixels same intensity).

Low value: More variation/randomness.

Formula idea: Sum of squared GLCM values.

2. Contrast
What it measures: Local intensity variations — how different neighboring pixels are.

High value: Strong edges / high variation.

Low value: Smooth image with similar neighboring pixels.

Formula idea: Weighted sum of squared intensity differences.

3. Correlation
What it measures: How correlated the pixel intensities are in a neighborhood.

High value: Pixels are related in a predictable way.

Low value: More randomness.

4. Variance
What it measures: Spread of pixel-pair values around the mean (texture diversity).

High value: Wide variety in texture patterns.

Low value: More uniform texture.

5. Inverse Difference Moment (IDM) / Homogeneity
What it measures: Closeness of distribution to the GLCM diagonal (i.e., how similar neighboring pixels are).

High value: Smooth textures.

Low value: High contrast textures.

6. Sum Average
What it measures: Average of the sum of the intensity pairs (related to brightness patterns).

7. Sum Variance
What it measures: Variability of the sum of pixel pair intensities.

8. Sum Entropy
What it measures: Disorder/unpredictability based on summed intensities.

High value: Very complex, noisy texture.

Low value: Regular, repetitive pattern.

9. Entropy
What it measures: Randomness in texture patterns.

High value: Random, unpredictable textures.

Low value: Ordered patterns.

Note: Often used for tumor irregularity detection.

The rest I didnt really understand. I need a good explanation of these features.
'''
