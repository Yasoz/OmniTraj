# OmniTraj

This repository contains the code to reproduce the KDD submissions for the paper titled “Learning Generalized and Flexible Trajectory Models from Omni-Semantic Supervision”





Files in the Repository

main.py: The main script to train the model
utils files contains the following files:
- dataset.py: Defines the dataset class with torch, including, trajectory, topology, road and regions. 
- omni_semantic.py: Defines the OmniTraj model with the modality encoders.
- traj_encoder.py: Defines the trajectory encoder.
- topol_encoder.py: Defines the topology encoder.
- road_encoder.py: Defines the road encoder.
- region_encoder.py: Defines the region encoder.

python main.py

data directory contains the dataset used in the paper. The dataset is a subset dataset with 1000 trajectories.


