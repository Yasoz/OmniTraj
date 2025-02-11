import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from einops.layers.torch import Rearrange
from typing import Dict, Union
from types import SimpleNamespace

import json
import os

class TrajectoryEncoder(nn.Module):
    """
    Trajectory encoder based on Vision Transformer architecture.
    
    Args:
        config: Configuration for the encoder
        
    Attributes:
        config: Model configuration
        num_patches: Number of patches the input is divided into
        
    Example:
        >>> config = SimpleNamespace(
        ...     input_dim=2, seq_length=200, patch_size=5,
        ...     embed_dim=256, depth=6, num_heads=8, pool='cls'
        ... )
        >>> model = TrajectoryEncoder(config)
        >>> x = torch.randn(32, 200, 2)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([32, 256])
    """
    def __init__(self, config: Union[SimpleNamespace, Dict]):
        super().__init__()
        # Set default values
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        self.config = config
        # Validate config
        self._validate_config()
        # Build model
        self._build_model()
        # Initialize weights
        self._init_weights()
        
    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.config.pooling_strategy in ['cls', 'mean'], 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert self.config.seq_length % self.config.patch_size == 0, 'Sequence length must be divisible by patch size'
        
    def _build_model(self):
        """Build model architecture"""
        # Extract config parameters
        input_dim = self.config.input_dim
        seq_length = self.config.seq_length
        patch_size = self.config.patch_size
        embed_dim = self.config.embed_dim
        depth = self.config.depth
        num_heads = self.config.num_heads
        # get extra parameters
        mlp_ratio = getattr(self.config, 'mlp_ratio', 4.0)
        drop_rate = getattr(self.config, 'drop_rate', 0.1)
        attn_drop_rate = getattr(self.config, 'attn_drop_rate', 0.1)
        drop_path_rate = getattr(self.config, 'drop_path_rate', 0.1)
        patch_embed_type = getattr(self.config, 'patch_embed_type', 'linear')
        # Calculate number of patches
        self.num_patches = seq_length // patch_size
        
        # Patch embedding layer
        if patch_embed_type == 'linear':
            self.patch_embed = nn.Sequential(
                Rearrange('b (n p) d -> b n (p d)', p=patch_size),
                nn.Linear(patch_size * input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Dropout(drop_rate)
            )
        elif patch_embed_type == 'conv':
            self.patch_embed = nn.Sequential(
                nn.Conv1d(input_dim, embed_dim, 
                         kernel_size=patch_size, 
                         stride=patch_size),
                Rearrange('b c n -> b n c'),
                nn.LayerNorm(embed_dim),
                nn.Dropout(drop_rate)
            )
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm
            )
            for i in range(depth)
        ])
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        # Final normalization layer
        self.norm = nn.LayerNorm(embed_dim)
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize patch embedding
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.apply(self._init_weights_layer)
        
    def _init_weights_layer(self, m):
        """Initialize layer weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor, attention_mask = None) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
        Returns: Tensor of shape [batch_size, embed_dim]
        """
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add position embedding
        x = x + self.pos_embed
        # Apply transformer blocks
        x = self.blocks(x)
        # Apply final normalization
        x = self.norm(x)
        # Pool output
        x = x.mean(dim=1) if self.config.pooling_strategy == 'mean' else x[:, 0]
        return x
    
    def save_pretrained(self, save_path: str):
        """
        Save model and config to disk.
        """
        os.makedirs(save_path, exist_ok=True)
        # Convert config to dict for saving
        config_dict = {key: getattr(self.config, key) for key in dir(self.config)  if not key.startswith('_')}
        # Save config
        config_path = os.path.join(save_path, 'trajectory_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        # Save model weights
        model_path = os.path.join(save_path, 'trajectory_encoder.pt')
        
         # handle multi-GPU models
        if isinstance(self, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state_dict = self.module.state_dict()
        else:
            state_dict = self.state_dict()
        torch.save(state_dict, model_path)
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Union[SimpleNamespace, Dict]) -> 'TrajectoryEncoder':
        """
        Load pretrained model from disk using provided configuration.
        Args:
            model_path: Path to pretrained model weights
            config: Model configuration (SimpleNamespace or dict)
        Example:
            >>> config = SimpleNamespace(input_dim=2, seq_length=200, ...)
            >>> model = TrajectoryEncoder.from_pretrained('model.pt', config)
        """
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        model = cls(config)
        # Check if model path is a directory
        if os.path.isdir(model_path):
            weight_path = os.path.join(model_path, 'trajectory_encoder.pt')
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
    
    @classmethod
    def get_default_config(cls) -> SimpleNamespace:
        """Get default configuration."""
        return SimpleNamespace(**cls.default_config)
    
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
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() 
                                        if p.requires_grad),
        }
 