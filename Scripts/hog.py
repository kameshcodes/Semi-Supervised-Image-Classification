import cv2
import os
import pandas as pd
import logging as log
import shutil
from Scripts.utils import *


def compute_hog(image, cell_size=(8, 8), block_size=(2, 2), bins=9):
    """
    Compute Histogram of Oriented Gradients (HOG) features for the given image.

    Parameters:
        image (numpy.ndarray): The input image.
        cell_size (tuple): Size of each cell for HOG computation. Default is (8, 8).
        block_size (tuple): Size of each block for HOG computation. Default is (2, 2).
        bins (int): Number of bins for HOG computation. Default is 9.

    Returns:
        numpy.ndarray: Computed HOG features.
    """
    try:
        hog = cv2.HOGDescriptor(_winSize=(image.shape[1] // cell_size[1] * cell_size[1],
                                           image.shape[0] // cell_size[0] * cell_size[0]),
                                 _blockSize=(block_size[1] * cell_size[1],
                                             block_size[0] * cell_size[0]),
                                 _blockStride=(cell_size[1], cell_size[0]),
                                 _cellSize=(cell_size[1], cell_size[0]),
                                 _nbins=bins)
        return hog.compute(image)
    except Exception as e:
        log.error(f"Error computing HOG features: {e}")
        return None
    
def compute_hog_features(image_path, normalize, cell_size=(8, 8), block_size=(2, 2), bins=9):
    """
    Compute Pyramid Histogram of Oriented Gradients (PHOG) features for the given image.

    Parameters:
        image_path (str): Path to the image file.
        cell_size (tuple): Size of each cell for HOG computation. Default is (8, 8).
        block_size (tuple): Size of each block for HOG computation. Default is (2, 2).
        bins (int): Number of bins for HOG computation. Default is 9.
        normalize (bool): Whether to normalize the features. Default is True.

    Returns:
        numpy.ndarray: Computed PHOG features.
    """
    image = read_image(image_path)

    hog_features = compute_hog(image, cell_size, block_size, bins)

    if normalize:
        hog_features = normalize_features(hog_features)

    return hog_features.flatten()

def make_hog_features_csv(input, output_csv_path, normalize, bins=9, cell_size=(8, 8), block_size=(2, 2)):
    """
    Compute PHOG features for all images in a directory and save them to a CSV file.

    Parameters:
        directory_path (str): Path to the directory containing images.
        cell_size (tuple): Size of each cell for HOG computation. Default is (8, 8).
        block_size (tuple): Size of each block for HOG computation. Default is (2, 2).
        bins (int): Number of bins for HOG computation. Default is 9.
        normalize (bool): Whether to normalize the features. Default is True.
    """
    phog_data = []
    try:
        for root, dirs, files in os.walk(input):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(image_path))
                    phog_features = compute_hog_features(image_path, normalize, cell_size, block_size, bins)
                    if phog_features is not None:
                        phog_data.append(phog_features.tolist() + [label])

        columns =  [f'HOG_{i}' for i in range(len(phog_data[0]) - 1)] + ['Label']
        df = pd.DataFrame(phog_data, columns=columns)
        
        create_new_output_folder(output_csv_path)

        output_csv_path = os.path.join(output_csv_path, "preprocessed.csv")
        df.to_csv(output_csv_path, index=False)
        log.info(f"PHOG features computed and saved to {output_csv_path}")
        
    except Exception as e:
        log.error(f"Error making PHOG features CSV: {e}")
        
if __name__ == "__main__":
    pass