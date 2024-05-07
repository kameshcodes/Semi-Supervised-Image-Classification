import logging as log
from Scripts.create_csv import make_csv_from_folder
from Scripts.split_dataset import split_csv

log_file = r'logs\make_dataset.log'
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    log.StreamHandler(), 
    log.FileHandler('logs\make_dataset.log') 
])

if __name__ == "__main__":
    input_folder_path = r'data\raw\101_ObjectCategories'
    
    preprocessed_image_output_path = r'data\pre-processed\images'
    preprocessed_csv_output_path = r'data\pre-processed\csv'
    
    split_input_file_path =  r'data\pre-processed\csv\preprocessed.csv'
    split_output_folder_path = r'data\train_test_csv'
    
    
    
    num_categories=5
    normalize = False
    test_size = 0.1
    
    log.info('Make Dataset Script started.')
    

    make_csv_from_folder(input_folder=input_folder_path, 
                        output_image_path=preprocessed_image_output_path, 
                        output_csv_path=preprocessed_csv_output_path,
                        num_categories=num_categories, 
                        normalize=normalize, 
                        cell_size=(8, 8), 
                        block_size=(2, 2), 
                        bins=9, 
                        size=(128, 128)
                        )

    
    split_csv(input_csv_path = split_input_file_path, 
                     output_csv_path=split_output_folder_path, 
                     test_size=test_size, 
                     random_state=2)
    
    log.info('Make Dataset Script complete.\n')
