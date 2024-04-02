import __main__
import os
import cv2
import logging as log
import numpy as np
import shutil

def read_image(image_path):
    """
    Read an image from the given path.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image array.
    """
    try:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        log.error(f"Error reading image '{image_path}': {e}")
        return None
    
def normalize_features(features):
    """
    Normalize the given features.

    Parameters:
        features (numpy.ndarray): The features to be normalized.

    Returns:
        numpy.ndarray: Normalized features.
    """
    try:
        return features / np.sum(np.abs(features))
    except Exception as e:
        log.error(f"Error normalizing features: {e}")
        return None
    


def create_new_output_folder(output_image_path):
    """
    Create a new output folder or delete and recreate if it already exists.

    Parameters:
        output_image_path (str): Path to the output folder.

    Returns:
        None
    """
    output_image_path = os.path.abspath(output_image_path)

    if os.path.exists(output_image_path):
        shutil.rmtree(output_image_path)
        log.info(f"Deleted existing folder: '{output_image_path}'.")

    os.makedirs(output_image_path)
    log.info(f"Created new folder: '{output_image_path}'.")

if __name__ == __main__:
    pass