#  configuration
encoder_configs = {
    'enabled_encoders': ['trajectory', 'topology', 'road','region'],
    'freeze_encoders': [],
    'pretrained_path': 'path/to/pretrained/models/', # Path to pretrained models
    'save_path': 'path/to/save/models/', # Path to save models
    'trajectory': {
        'input_dim': 2,
        'seq_length': 200,
        'patch_size': 5,
        'embed_dim': 256,
        'depth': 6,
        'num_heads': 8,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.1,
        'drop_path_rate': 0.1,
        'pooling_strategy': 'cls',
        'patch_embed_type': 'linear'
    },
    'topology': {
        'input_dim': 2,
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'max_position_embeddings': 256,
        'pooling_strategy': 'cls'
    },
    'road': {
        'num_roads': 8000,
        'output_dim': 256,
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'pooling_strategy': 'cls'
    },
    "region": {
        "num_grids": 256,
        "output_dim": 256,
        "embedding_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "dropout": 0.1,
        "pooling_strategy": "cls"
    }
}

dataset_configs = {
    "chengdu": {
        "dataset_params": {
            "dataset_name": "chengdu",
            "topol_len": 128,
            "traj_len": 200,
            "num_grids": 256,
            "num_roads": 8000,
            "max_road_length": 128,
            "max_region_length": 64,
            "transform": True,
            "augment": True,
        },
        "loader_params": {
            "train": {
                "batch_size": 1536,
                "shuffle": True,
                "num_workers": 24,
                "pin_memory": True
            },
            "valid": {
                "batch_size": 1024,
                "shuffle": False,
                "num_workers": 8,
                "pin_memory": True
            }
        }
    },
    "xian": {
        "dataset_params": {
            "dataset_name": "xian",
            "topol_len": 128,
            "traj_len": 200,
            "num_grids": 256,
            "num_roads": 8000,
            "max_road_length": 128,
            "max_region_length": 64,
            "transform": True,
            "augment": True,
        },
        "loader_params": {
            "train": {
                "batch_size": 1536,
                "shuffle": True,
                "num_workers": 24,
                "pin_memory": True
            },
            "valid": {
                "batch_size": 1024,
                "shuffle": False,
                "num_workers": 8,
                "pin_memory": True
            }
        }
    }
}
