import cv2
import os
import glob
import shutil
import logging as log

#################################################################################################
from Scripts.cnn import make_cnn_features_csv
#from Scripts.hog import make_hog_features_csv
################################################################################################
from Scripts.utils import create_new_output_folder
     
def preprocess_image(image_path, output_path, size=(128, 128)):
    """
    Preprocess an image by resizing and converting to grayscale.

    Parameters:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the preprocessed image.
        size (tuple): Desired size of the output image. Default is (128, 128).
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            log.warning(f"Could not read image '{image_path}'.")
            return

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, size)
        
        #resized_image = cv2.resize(image, size)
        cv2.imwrite(output_path, resized_image)
    except Exception as e:
        log.error(f"Error preprocessing image '{image_path}': {e}")

def make_csv_from_folder(input_folder, output_image_path, output_csv_path, num_categories, normalize, cell_size=(8, 8), block_size=(2, 2), bins=9, size=(128, 128)):
    """
    Preprocess images in a folder and save them to another folder.

    Parameters:
        input_folder (str): Path to the input folder containing images.
        output_image_path (str): Path to the output folder to save preprocessed images.
        size (tuple): Desired size of the output images. Default is (128, 128).
        num_categories (int): Number of categories to process. Default is 5.
    """
    try:
        log.info(f"Preprocessing images from {input_folder}")
        input_folder = os.path.abspath(input_folder)
        
        create_new_output_folder(output_image_path)

        # count image in each category
        category_counts = {}
        for root, dirs, files in os.walk(input_folder):
            for dir in dirs:
                category_counts[dir] = len(glob.glob(os.path.join(root, dir, '*.[jJ][pP][gG]')))
        
        # sort categories
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:num_categories]

        for category, _ in sorted_categories:
            category_folder = os.path.join(input_folder, category)
            output_category_folder = os.path.join(output_image_path, category)
            os.makedirs(output_category_folder, exist_ok=True)

            for image_path in glob.glob(os.path.join(category_folder, '*.[jJ][pP][gG]')):
                output_image_folder_path = os.path.join(output_category_folder, os.path.basename(image_path))
                preprocess_image(image_path, output_image_folder_path, size)

        log.info(f"Images preprocessed and saved in: '{output_image_path}'.")
        
        ###########################################################################################################################
        #make_hog_features_csv(output_image_path, output_csv_path, normalize, bins, cell_size, block_size)
        make_cnn_features_csv(output_image_path, output_csv_path)
        ###########################################################################################################################
        
        
    except Exception as e:
        log.error(f"Error preprocessing images: {e}")
        
    
if __name__ == "__main__":
    pass
