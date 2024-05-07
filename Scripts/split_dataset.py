import os 
import pandas as pd
import logging as log
from sklearn.model_selection import train_test_split
from Scripts.utils import create_new_output_folder

def split_csv(input_csv_path, output_csv_path, test_size, random_state=2):
    """
    Split the preprocessed CSV file into stratified train and test sets.

    Parameters:
        input_csv_path (str): Path to the preprocessed CSV file.
        train_csv_path (str): Path to save the train CSV file.
        test_csv_path (str): Path to save the test CSV file.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is None.
    """
    try:
        df = pd.read_csv(input_csv_path)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Label'])
        
        ##################################################################################################
        # train_df.to_csv(os.path.join(output_csv_path, "train_data_hog.csv"), index=False)
        # test_df.to_csv(os.path.join(output_csv_path, "test_data_hog.csv"), index=False)
        
        train_df.to_csv(os.path.join(output_csv_path, "train_data_conv.csv"), index=False)
        test_df.to_csv(os.path.join(output_csv_path, "test_data_conv.csv"), index=False)
        ##################################################################################################

        log.info(f"Train Size: '{len(train_df)}, Test Size {len(test_df)}'.")
        log.info(f"Train and test sets saved to '{output_csv_path}'.")

    except Exception as e:
        log.error(f"Error occurred: {e}")

if __name__ == "__main__":
    pass
