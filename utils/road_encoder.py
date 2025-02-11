import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from transformers import RoFormerModel, RoFormerConfig
from typing import Dict, Union
from types import SimpleNamespace
import os
import json

class RoadEncoder(nn.Module):
    """
    Road encoder based on RoFormer architecture.
    
    Args:
        config: Configuration for the encoder
        
    Attributes:
        config: Model configuration
        PAD_IDX: Padding token index
    Example:
        >>> config = SimpleNamespace(
        ...     num_roads=8000,
        ...     embed_dim=256,
        ...     output_dim=256,
        ...     num_heads=8,
        ...     depth=6,
        ...     pool='bos',
        ...     max_position_embeddings=256
        ... )
        >>> model = RoadEncoder(config)
        >>> x = torch.randint(0, 8000, (32, 128))  # [batch_size, seq_length]
        >>> attention_mask = torch.ones_like(x)
        >>> output = model(x, attention_mask)
        >>> print(output.shape)  # [32, 256]
    """
    def __init__(self, config: Union[SimpleNamespace, Dict]):
        super().__init__()
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        self.config = config
        # Special tokens
        self.PAD_IDX = 0
        # Validate config
        self._validate_config()
        # Build model
        self._build_model()
        # Initialize weights
        self.apply(self._init_weights)
        
    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.config.pooling_strategy in ['mean', 'cls', 'bos', 'max'], 'pool type must be either mean, cls, bos or max'
        assert hasattr(self.config, 'num_roads'), 'num_roads must be specified in config'
        
    def _build_model(self):
        """Build model architecture"""
        # Extract config parameters
        num_roads = self.config.num_roads
        embed_dim = getattr(self.config, 'embed_dim', 256)
        output_dim = getattr(self.config, 'output_dim', embed_dim)
        num_heads = getattr(self.config, 'num_heads', 8)
        n_layers = getattr(self.config, 'num_layers', 6)
        dropout = getattr(self.config, 'dropout', 0.1)
        max_position_embeddings = getattr(self.config, 'max_position_embeddings', 256)
        
        # Road embedding layer
        self.road_embedding = nn.Embedding(
            num_embeddings=num_roads,
            embedding_dim=embed_dim,
            padding_idx=self.PAD_IDX
        )
        
        # Projection layer if needed
        self.projection = nn.Linear(embed_dim, output_dim) if embed_dim != output_dim else nn.Identity()
        
        # Configure RoFormer
        roformer_config = RoFormerConfig(
            vocab_size=1, # not used just for consistency with RoFormer 
            hidden_size=output_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=n_layers,
            intermediate_size=output_dim*2,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings,
            rotary_value=True,
            use_cache=False
        )
        
        # RoFormer encoder
        self.roformer = RoFormerModel(roformer_config)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
    def _pool_output(self, last_hidden_state, attention_mask):
        """Pool the output according to the specified strategy"""
        if self.config.pooling_strategy == "cls" or self.config.pool == "bos":
            return last_hidden_state[:, 0]
        elif self.config.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        else:  # max
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            last_hidden_state = last_hidden_state.masked_fill(~mask_expanded.bool(), float('-inf'))
            return torch.max(last_hidden_state, dim=1)[0]
            
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape [batch_size, seq_length] containing road IDs
            attention_mask: Optional attention mask tensor of shape [batch_size, seq_length]
        Returns: 
            Tensor of shape [batch_size, output_dim]
        """
        # 1. Road embedding
        hidden_state = self.road_embedding(x)
        
        # 2. Project if needed
        hidden_state = self.projection(hidden_state)
        
        # 3. RoFormer encoding
        roformer_outputs = self.roformer(
            inputs_embeds=hidden_state,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 4. Pool and return output
        output = self._pool_output(roformer_outputs.last_hidden_state, attention_mask)
        
        return output
    
    def save_pretrained(self, save_path: str):
        """Save model and config to disk."""
        os.makedirs(save_path, exist_ok=True)
        # Convert config to dict for saving
        config_dict = {key: getattr(self.config, key) for key in dir(self.config) if not key.startswith('_')}
        # Save config
        config_path = os.path.join(save_path, 'road_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        # Save model weights
        model_path = os.path.join(save_path, 'road_encoder.pt')
        
        # Handle multi-GPU models
        if isinstance(self, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state_dict = self.module.state_dict()
        else:
            state_dict = self.state_dict()
        torch.save(state_dict, model_path)
        
    @classmethod
    def from_pretrained(cls, model_path: str, config: Union[SimpleNamespace, Dict]) -> 'RoadEncoder':
        """
        Load pretrained model from disk using provided configuration.
        Args:
            model_path: Path to pretrained model weights
            config: Model configuration (SimpleNamespace or dict)
        Returns:
            Loaded model instance
        """
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        model = cls(config)
        # Check if model path is a directory
        if os.path.isdir(model_path):
            weight_path = os.path.join(model_path, 'road_encoder.pt')
        else:
            weight_path = model_path
        # Load model weights
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"No model weights found at {weight_path}")
        try:
            state_dict = torch.load(weight_path, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
        return model
    
    def get_model_info(self) -> Dict:
        """
        Get model information including configuration and parameters.
        Returns:
            Dictionary containing model information
        """
        return {
            'config': {key: getattr(self.config, key) 
                    for key in dir(self.config) if not key.startswith('_')},
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters()  if p.requires_grad),
        }
