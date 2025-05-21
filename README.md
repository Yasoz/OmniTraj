# Learning Generalized and Flexible Trajectory Models from Omni-Semantic Supervision

## Overview
OmniTraj is a novel multi-modal trajectory representation learning framework that leverages multiple semantic modalities (trajectory, topology, road network, and region) to learn generalized and flexible trajectory models. This repository contains the implementation of our KDD'2025 paper.

## Features
- Multi-modal trajectory retrieval
- Support for four semantic modalities:
  - Trajectory: Raw GPS points
  - Topology: Topological structure of trajectories
  - Road: Road network segments
  - Region: Grid-based region representation
- Flexible model architecture with modality-specific encoders
- Contrastive learning framework for cross-modal alignment



## Project Structure
```
.
├── data/                     # Directory for storing datasets
│   └── trajectory.pkl        # Example dataset file (replace with your actual data)
├── utils/                    # Utility scripts and model components
│   ├── omni_semantic.py      # Defines the core OmniModel
│   ├── trajectory_encoder.py # Implements the TrajectoryEncoder
│   ├── topology_encoder.py   # Implements the TopologyEncoder
│   ├── road_encoder.py       # Implements the RoadEncoder
│   ├── region_encoder.py     # Implements the RegionEncoder
│   ├── dataset.py            # Handles data loading and preprocessing (TrajectoryDataset)
│   ├── config.py             # Contains model and dataset configurations
│   └── utils.py              # General utility functions (e.g., AvgMeter)
├── main.py                   # Main script for training and evaluation
├── requirements.txt          # Python dependencies for the project
└── README.md                 # This file
```
## Prerequisites and Installation

To set up and run this project, you'll need the following:

*   Python 3 (tested with Python 3.8 and newer)
*   PyTorch (ensure it's compatible with your CUDA version if using GPU)

We recommend using a virtual environment to manage dependencies:

## Model Architecture
OmniTraj consists of four modality-specific encoders:

1. **Trajectory Encoder**: 
   - Processes raw GPS points (optional for linear interpolation)
   - Uses patch embedding and transformer blocks

2. **Topology Encoder**:
   - Handles topological structure of trajectories
   - Captures spatial relationships between points

3. **Road Encoder**:
   - Processes road network segments
   - Maps road segments to embeddings

4. **Region Encoder**:
   - Grid-based region representation
   - Captures spatial distribution

The model uses a contrastive learning framework to align different modalities and learn a unified representation.

## Usage Instructions

### 1. Data Preparation

1. **Dataset**: This project expects trajectory data in a `pkl` file format, as exemplified by `data/trajectory.pkl`. Each entry in the pickle file should contain the necessary trajectory information, including sequences for `trajectory`, `topology`, `roads`, and `cell_sequence`.

2. **Configuration**:

   1. Key training parameters can be modified in `utils/config.py`:

      - Model architecture parameters
      - Training hyperparameters
      - Dataset configurations
      - Data augmentation settings

   2. Place your dataset (e.g., `your_dataset_train.pkl`, `your_dataset_val.pkl`) into a suitable directory, for example, `../Trajdata/` relative to the project root, or update the paths directly in `main.py`.

   3. In `main.py`, within the `if __name__ == "__main__":` block, modify the `dataset_name` variable (e.g., 'xian', 'chengdu') to match your dataset key in `utils/config.py`.

   4. The paths for training and testing files are constructed based on this `dataset_name`:

      ```python
      dataset_name ='xian' # Or 'chengdu', or your custom dataset key
      path = {
          "train_file_path": f'../Trajdata/{dataset_name}_train.pkl',
          "test_file_path": f'../Trajdata/{dataset_name}_val.pkl'
      }
      config = dataset_configs[dataset_name]
      config.update(path)
      ```

   5. Ensure that `dataset_configs[dataset_name]` in `utils/config.py` correctly defines parameters for your data.


### 2. Running Training

Once the data is prepared and paths are configured:

```bash
python main.py
```

The script will:

*   Set up experiment directories under `OmniModel/<dataset_name>-<timestamp>/`.
*   Save a copy of the `utils` scripts and `main.py` into `OmniModel/<dataset_name>-<timestamp>/Files/` for reproducibility.
*   Train the `OmniModel` using the specified training data.
*   Perform validation on the test data after each epoch.
*   Save the best performing model (based on validation loss) to `OmniModel/<dataset_name>-<timestamp>/models/best_OmniModel.pt`.
*   Periodically save model checkpoints (e.g., every 20 epochs) to the same `models` directory.




## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{zhu2025learning,
    title={Learning Generalized and Flexible Trajectory Models from Omni-Semantic Supervision},
    author={Zhu, Yuanshao and Yu, James Jianqiao and Zhao, Xiangyu and Han, Xiao and Liu, Qidong and Wei, Xuetao and Liang, Yuxuan},
    booktitle={Proceedings of the 31th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    year={2025}
}
```
