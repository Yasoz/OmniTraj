# OmniTraj

This repository contains the code to reproduce the KDD submissions for the paper titled “Learning Generalized and Flexible Trajectory Models from Omni-Semantic Supervision”

Requirements

The code is implemented in Python and uses the PyTorch framework. To ensure compatibility, please install the following dependencies:

Files in the Repository

unitraj.py: Defines the Unitraj model architecture, and other core functionalities.
dataset.py: Defines the dataset class, including resampling, masking strategies and corrsponding functions.
main.py: The primary script for training the Unitraj model.
config.py: Contains the configuration parameters for the model.
you can run the code by running the following command:

python main.py
Data

We provide a sample dataset in the data folder. The dataset is a subset dataset with 1000 trajectories. Each trajectory contains trajectory points, and timestamps. The dataset is stored in a .pkl.

We also provide data load notebook to load the data and visualize the trajectories. Saved as load_see_data.ipynb

Model

We provide the trained model in the folder. The model is saved as a .pt file.
