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

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU support)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/OmniTraj.git
cd OmniTraj
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
OmniTraj/
├── main.py                 # Main training script
├── requirements.txt        # Project dependencies
├── data/                   # Dataset directory
│   ├── trajectory.pkl  # Sample of Chengdu dataset
│   ├── 
└── utils/                  # Utility modules
    ├── dataset.py         # Dataset class implementation
    ├── omni_semantic.py   # OmniTraj model definition
    ├── traj_encoder.py    # Trajectory encoder
    ├── topol_encoder.py   # Topology encoder
    ├── road_encoder.py    # Road network encoder
    ├── region_encoder.py  # Region encoder
    ├── config.py          # Configuration settings
    └── utils.py           # Utility functions
```

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

## Usage

### Training
To train the model on the Chengdu dataset:
```bash
python main.py
```


### Configuration
Key training parameters can be modified in `utils/config.py`:
- Model architecture parameters
- Training hyperparameters
- Dataset configurations
- Data augmentation settings

### Model Checkpoints
- Best models are saved in the `models/` directory
- Checkpoints are saved every 20 epochs
- Best model is saved based on validation loss




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
