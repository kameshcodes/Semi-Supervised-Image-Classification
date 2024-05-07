import cv2
import os
import pandas as pd
import logging as log
import shutil
from Scripts.utils import read_image, normalize_features, create_new_output_folder
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def compute_cnn_features(image_path, input_shape=(128, 128, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Use Input layer
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
    ])
    image = read_image(image_path) 
    features = model.predict(tf.expand_dims(image, axis=0)) 
    flat = features.flatten()
    return flat

def make_cnn_features_csv(input_folder_path, output_csv_path):
    cnn_data = []
    try:
        for root, dirs, files in os.walk(input_folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(image_path))
                    cnn_features = compute_cnn_features(image_path)
                    if cnn_features is not None:
                        cnn_data.append(cnn_features.tolist() + [label])

        columns = [f'Conv_feat_{i}' for i in range(len(cnn_data[0]) - 1)] + ['Label']
        df = pd.DataFrame(cnn_data, columns=columns)
        create_new_output_folder(output_csv_path)
        output_csv_path = os.path.join(output_csv_path, "preprocessed.csv")
        df.to_csv(output_csv_path, index=False)
        
        log.info(f"CNN features computed and saved to {output_csv_path}")
        
    except Exception as e:
        log.error(f"Error making CNN features CSV: {e}")

if __name__ == "__main__":
    pass