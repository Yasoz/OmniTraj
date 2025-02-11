import torch
from torch import nn
import torch.nn.functional as F
import os
from typing import Dict, Optional, Union, List, Tuple
from types import SimpleNamespace

from .road_encoder import RoadEncoder
from .region_encoder import RegionEncoder
from .topol_encoder import TopologyEncoder
from .traj_encoder import TrajectoryEncoder


class OmniModel(nn.Module):
    """
    Contrastive Language-Image Pre-training model for trajectory retrieval.
    Args:
        encoder_configs: Configuration containing model parameters for each encoder
        contrast_pairs: List of tuples specifying which modalities to contrast
        projection_dim: Output dimension for projection heads
    """
    ENCODER_CLASSES = {
        "trajectory": TrajectoryEncoder,
        "topology": TopologyEncoder,
        "road": RoadEncoder,
        "region": RegionEncoder
    }
    def __init__(
        self,
        encoder_configs: Dict[str, dict],
        contrast_pairs: List[Tuple[str, str]],
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(encoder_configs, dict):
            encoder_configs = SimpleNamespace(**encoder_configs)
        self.config = encoder_configs
        # Initialize basic components
        self.encoders = nn.ModuleDict()
        self.projection_heads = nn.ModuleDict()
        self.contrast_pairs = contrast_pairs
        self.fusion_layers = nn.ModuleDict()
        
        # Get enabled encoders
        enabled_encoders = getattr(self.config, 'enabled_encoders', ['trajectory', 'topology', 'road', 'region'])
        freeze_encoders = getattr(self.config, 'freeze_encoders', [])
        pretrained_path = getattr(self.config, 'pretrained_path', None)
        
        # Initialize enabled encoders
        for encoder_name in enabled_encoders:
            if encoder_name not in self.ENCODER_CLASSES:
                raise ValueError(f"Unknown encoder type: {encoder_name}")
            # Get encoder specific configs
            encoder_cfg = getattr(self.config, encoder_name, None)
            if encoder_cfg is None:
                raise ValueError(f"Missing configuration for encoder: {encoder_name}")
            # Initialize encoder
            self.encoders[encoder_name] = self.ENCODER_CLASSES[encoder_name](encoder_cfg)
            
            # Initialize projection head
            input_dim = encoder_cfg.output_dim if hasattr(encoder_cfg, 'output_dim') else encoder_cfg.embed_dim
            if projection_dim is None:
                projection_dim = input_dim *2
            
            self.projection_heads[encoder_name] = nn.Sequential(
                nn.Linear(input_dim, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
                nn.Dropout(0.1),
                nn.LayerNorm(projection_dim)
            )
            
        self.fusion_layers = nn.ModuleDict({
            'topology_road': nn.Linear(projection_dim*2, projection_dim),
            'topology_region': nn.Linear(projection_dim*2, projection_dim),
            'road_region': nn.Linear(projection_dim*2, projection_dim),
            'topology_road_region': nn.Linear(projection_dim*3, projection_dim)
        })
        # set fusion_
        # Initialize temperature parameter
        self.temperature = nn.Parameter(torch.ones([]) * 1.0)
        
        self._init_weights()
        
        
        # load pretrained models if specified
        if freeze_encoders and pretrained_path:
            for encoder_name in freeze_encoders:
                self.load_encoder(encoder_name, os.path.join(pretrained_path, f'{encoder_name}.pt'))
        
                self._freeze_module(self.encoders[encoder_name])
                self._freeze_module(self.projection_heads[encoder_name])
                print(f"Loaded and frozen {encoder_name} encoder and projection head")

    
    def _init_weights(self):
        """Initialize projection heads and fusion layers weights"""
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for head in self.projection_heads.values():
            head.apply(_init_module)
        for fusion in self.fusion_layers.values():
            fusion.apply(_init_module)
            
    def forward(self, batch: Dict[str, torch.Tensor], fusion_modality: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass for contrastive learning
        Args:
            batch: Dictionary containing modality inputs and attention masks
        Returns:
            Dictionary containing losses for each contrast pair and total loss
            # topols = batch['topology']
            # roads = batch['road']
            # regions = batch['region']
            # topols_mask = batch.get('topology_attention_mask', None)
            # roads_mask = batch.get('road_attention_mask', None)
            # regions_mask = batch.get('region_attention_mask', None)
        """
        embeddings = {}
        # Compute embeddings for all required modalities
        for modality in set(sum(self.contrast_pairs, ())):
            embeddings[modality] = self.encode_modality(
                modality,
                batch[modality],
                batch.get(f'{modality}_attention_mask', None),
                normalize=False
            )
        # Compute fusion embeddings if needed
        if fusion_modality:
            for fusion_type in fusion_modality:
                fused_embed = self.encode_fusion(fusion_type, embeddings, normalize=False)
                embeddings[fusion_type] = fused_embed
        
        # Compute losses for each contrast pair
        losses = {}
        total_loss = 0.0
        for mod1, mod2 in self.contrast_pairs:
            pair_loss = self.compute_contrastive_loss(
                embeddings[mod1], 
                embeddings[mod2]
            )
            losses[f"{mod1}_{mod2}"] = pair_loss.mean()
            total_loss += pair_loss.mean()
            
        if fusion_modality:
            for fusion_type in fusion_modality:
                pair_loss = self.compute_contrastive_loss(
                    embeddings['trajectory'], 
                    embeddings[fusion_type]
                )
                losses[f"{'trajectory'}_{fusion_type}"] = pair_loss.mean()
                total_loss += pair_loss.mean()

        # Compute average loss
        # losses['total_loss'] = sum(losses.values()) / len(losses)
        loss = total_loss / len(losses)
        return loss
    
    def encode_modality(self, modality: str, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, normalize: bool = False) -> torch.Tensor:
        """Generic encoding method for any modality"""
        if modality not in self.encoders:
            raise ValueError(f"Unknown modality: {modality}")
        features = self.encoders[modality](x, attention_mask)
        embeddings = self.projection_heads[modality](features)
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def encode_fusion(self, fusion_type: str, features: Dict[str, torch.Tensor], normalize: bool = False) -> torch.Tensor:
        """Encode fused features from multiple modalities"""
        if fusion_type not in self.fusion_layers:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        if fusion_type == 'topology_road':
            fused = torch.cat([features['topology'], features['road']], dim=-1)
        elif fusion_type == 'topology_region':
            fused = torch.cat([features['topology'], features['region']], dim=-1)
        elif fusion_type == 'road_region':
            fused = torch.cat([features['road'], features['region']], dim=-1)
        elif fusion_type == 'topology_road_region':
            fused = torch.cat([features['topology'], features['road'], features['region']], dim=-1)
        
        embeddings = self.fusion_layers[fusion_type](fused)
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    

    @torch.no_grad()
    def get_embeddings(self, batch):
        """get embeddings for all modalities"""
        embeddings = {}
        # encodding the basic modalities
        for modality in self.encoders:
            embeddings[modality] = self.encode_modality(
                modality,
                batch[modality],
                batch.get(f'{modality}_attention_mask'),
                normalize=True
            )
        if hasattr(self, 'fusion_layers'):
            for fusion_type in self.fusion_layers:
                embeddings[fusion_type] = self.encode_fusion(
                    fusion_type, 
                    embeddings,
                    normalize=True
                )
        return embeddings
    
    def compute_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two embeddings"""
        # Compute similarity matrices
        similarity = torch.matmul(z1, z2.T) / self.temperature
        sim_1 = torch.matmul(z1, z1.T)
        sim_2 = torch.matmul(z2, z2.T)
        # Compute soft targets
        targets = F.softmax((sim_1 + sim_2) / 2 * self.temperature, dim=-1)
        # Compute losses
        loss_1 = self.cross_entropy(similarity, targets,reduction='none')
        loss_2 = self.cross_entropy(similarity.T, targets.T,reduction='none')
        return (loss_1 + loss_2) / 2

    def cross_entropy(self, preds: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute cross entropy loss"""
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "mean":
            return loss.mean()
        return loss


              
    def load_encoder(self, encoder_name: str, load_path: str):
        """Load encoder state dict and configuration"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No pretrained weights found at {load_path}")
        try:
            encoder_state = torch.load(load_path, weights_only=True)
            self.encoders[encoder_name].load_state_dict(encoder_state['encoder'])
            self.projection_heads[encoder_name].load_state_dict(encoder_state['projection'])
            print(f"Successfully loaded weights for {encoder_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load encoder weights: {str(e)}")
        
    def _freeze_module(self, module):
        """Freeze parameters of a module"""
        for param in module.parameters():
            param.requires_grad = False
            
    def _unfreeze_module(self, module):
        """Unfreeze parameters of a module"""
        for param in module.parameters():
            param.requires_grad = True

    def freeze_encoder(self, encoder_name: str):
        """Freeze specific encoder and its projection head"""
        if encoder_name in self.encoders:
            self._freeze_module(self.encoders[encoder_name])
            self._freeze_module(self.projection_heads[encoder_name])

    def unfreeze_encoder(self, encoder_name: str):
        """Unfreeze specific encoder and its projection head"""
        if encoder_name in self.encoders:
            self._unfreeze_module(self.encoders[encoder_name])
            self._unfreeze_module(self.projection_heads[encoder_name])
        
    def save_encoder(self, encoder_name: str, save_path: str):
        """Save encoder state dict and configuration"""
        if encoder_name in self.encoders:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            encoder_state = {
                'encoder': self.encoders[encoder_name].state_dict(),
                'projection': self.projection_heads[encoder_name].state_dict(),
                'config': getattr(self.config, encoder_name)
            }
            torch.save(encoder_state, save_path) 
            
    def get_model_info(self) -> Dict:
        """Get model information including parameters and configurations"""
        return {
            'enabled_encoders': list(self.encoders.keys()),
            'contrast_pairs': self.contrast_pairs,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() 
                                         if p.requires_grad),
            'encoder_configs': self.encoder_configs
        }
        

    def model_summary(self) -> Dict:
        """Return model summary information"""
        return {
            'enabled_encoders': list(self.encoders.keys()),
            'contrast_pairs': self.contrast_pairs,
            'fusion_layers': list(self.fusion_layers.keys()) if hasattr(self, 'fusion_layers') else [],
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'device': next(self.parameters()).device
        }
        
    def save_checkpoint(self, path: str):
        """Save full model checkpoint"""
        checkpoint = {
            'model_state': self.state_dict(),
            'config': self.config,
            'contrast_pairs': self.contrast_pairs,
            'temperature': self.temperature.item()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load full model checkpoint"""
        checkpoint = torch.load(path,weights_only=True)
        self.load_state_dict(checkpoint['model_state'])
        self.contrast_pairs = checkpoint['contrast_pairs']
        self.temperature.data = torch.tensor(checkpoint['temperature'])
    