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

class TopologyEncoder(nn.Module):
    """
    Topology encoder based on RoFormer architecture.
    
    Args:
        config: Configuration for the encoder
        
    Attributes:
        config: Model configuration
        
    Example:
        >>> config = SimpleNamespace(
        ...     input_dim=64,
        ...     embed_dim=256,
        ...     num_heads=8,
        ...     depth=6,
        ...     pool='cls',
        ...     max_position_embeddings=256
        ... )
        >>> model = TopologyEncoder(config)
        >>> x = torch.randn(32, 128, 2)
        >>> attention_mask = torch.ones(32, 128)
        >>> output = model(x, attention_mask)
        >>> print(output.shape)
        torch.Size([32, 256])
    """
    def __init__(self, config: Union[SimpleNamespace, Dict]):
        super().__init__()
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        self.config = config
        # Validate config
        self._validate_config()
        # Build model
        self._build_model()
        # Initialize weights
        self.apply(self._init_weights)
        
    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.config.pooling_strategy in ['cls', 'mean', 'max'], 'pool type must be either cls, mean or max'
        
    def _build_model(self):
        """Build model architecture"""
        # Extract config parameters
        input_dim = self.config.input_dim
        embed_dim = self.config.embed_dim
        num_heads = self.config.num_heads
        n_layers = self.config.num_layers
        # Get extra parameters
        dropout = getattr(self.config, 'dropout', 0.1)
        max_position_embeddings = getattr(self.config, 'max_position_embeddings', 256)
        
        # Configure RoFormer
        roformer_config = RoFormerConfig(
            vocab_size=1, # not used just for consistency with RoFormer 
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=n_layers,
            intermediate_size=embed_dim*4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings,
            rotary_value=True,
            use_cache=False
        )
        
        # Input projection with LayerNorm and Dropout
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # RoFormer encoder
        self.roformer = RoFormerModel(roformer_config)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def _pool_output(self, last_hidden_state, attention_mask):
        """
        Pool the output according to the specified strategy.
        Args:
            last_hidden_state: Tensor of shape [batch_size, seq_length, embed_dim]
            attention_mask: Tensor of shape [batch_size, seq_length]
        Returns:
            Pooled output tensor of shape [batch_size, embed_dim]
        """
        if self.config.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.config.pooling_strategy == "cls":
            return last_hidden_state[:, 0]
        else:  # max
            # Use attention_mask to ensure padding positions are not selected
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            last_hidden_state = last_hidden_state.masked_fill(~mask_expanded.bool(), float('-inf'))
            return torch.max(last_hidden_state, dim=1)[0]
            
    def forward(self, x: torch.Tensor, attention_mask) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            attention_mask: attention mask tensor of shape [batch_size, seq_length]
        Returns: 
            Tensor of shape [batch_size, embed_dim]
        """
        batch_size = x.size(0)
        # 1. Feature projection
        hidden_state = self.input_projection(x)
        
        # 2. Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        hidden_state = torch.cat([cls_tokens, hidden_state], dim=1)
        
        # 3. Update attention mask
        cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # 4. RoFormer encoding
        roformer_outputs = self.roformer(
            inputs_embeds=hidden_state,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 5. Pool and return output
        output = self._pool_output(roformer_outputs.last_hidden_state, attention_mask)
        
        return output
    
    def save_pretrained(self, save_path: str):
        """
        Save model and config to disk.
        """
        os.makedirs(save_path, exist_ok=True)
        # Convert config to dict for saving
        config_dict = {key: getattr(self.config, key) for key in dir(self.config) if not key.startswith('_')}
        # Save config
        config_path = os.path.join(save_path, 'topology_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        # Save model weights
        model_path = os.path.join(save_path, 'topology_encoder.pt')
        
        # Handle multi-GPU models
        if isinstance(self, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state_dict = self.module.state_dict()
        else:
            state_dict = self.state_dict()
        torch.save(state_dict, model_path)
        
    @classmethod
    def from_pretrained(cls, model_path: str, config: Union[SimpleNamespace, Dict]) -> 'TopologyEncoder':
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
            weight_path = os.path.join(model_path, 'topology_encoder.pt')
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
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() 
                                        if p.requires_grad),
        }