# Course Project - IE506: Machine Learning Principle and Techniques

---

## Topic - Semi-Supervised Random Forests

---

### Info

- **Title**: Semi-Supervised Random Forests Implementation
- **Team**: Phoenix_22N0088
- **Members**:
  - **Name**: Kamesh Dubey
  - **Roll No**: 22N0088

## Steps to Clone and Set Up the Project

---

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/kameshcodes/Semi-Supervised-Image-Classification.git
    ```

2. **Navigate to the Project Directory**:

    After cloning the repository, you'll need to access the local folder created on your computer. To do so, execute the following command in your terminal:

    ```bash
    cd Semi-Supervised-Image-Classification

    ```

3. **Create a Virtual Environment**:

    ```bash
    python -m venv myvenv
    ```

    This command creates a virtual environment named `myvenv` in the project directory.

4. **Activate the Virtual Environment**:

    - On Windows:

      ```bash
      myvenv\Scripts\activate
      ```

    - On macOS and Linux:

      ```bash
      source myvenv/bin/activate
      ```

5. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

---

For this project, the Caltech 101 dataset was utilized. The Caltech 101 dataset is a widely used benchmark dataset in computer vision, containing images of objects belonging to 101 categories.

- Dataset Website: [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02)

To prepare the dataset for use in the project, it needs to undergo preprocessing and splitting into training and testing subsets. This involves tasks such as resizing images, normalization, and partitioning the data into sets for training the model and evaluating its performance.

To carry out these preprocessing and splitting steps, the following command can be executed:

```bash
python make_dataset.py
```

### Directory Hierarchy

```bash

|—— artifacts
|—— data
|   |—— preprocessed
|   |—— raw
|   |—— test_data
|   |—— train_data
|—— logs
|   |—— make_dataset.log
|—— models
|—— notebooks
|—— Scripts
|   |—— preprocess.py
|   |—— split_dataset.py
|—— .gitignore
|—— make_dataset.py
```

## Code Details

The project's file structure is organized as follows:

- `data/`: Directory for storing raw, preprocessed, and split data.
- `models/`: Stores trained machine learning models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and experimentation.
- `scripts/`: Contains Python scripts for data preprocessing, dataset splitting, model training, and evaluation. This directory contains the main project files.
- `make_dataset.py`: Script creating preporcess images and form csvfile and splitting csv_file into train-test.
- `train_model.py` :

### Tested Platform

- software
  - Python: 3.11.5
  - OS: Windows 11 Home Single Language

- hardware
  - CPU: AMD Ryzen 5 5625U with Radeon Graphics

## References

- [Python Documentation](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [OpenCV Documentation](https://docs.opencv.org/master/)

## Citing

```bibtex
@article{leistner2009semi,
  title={Semi-Supervised Random Forests},
  author={Leistner, Christian and Saffari, Amir and Santner, Jakob and Bischof, Horst},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2009},
  publisher={IEEE}
}
```
