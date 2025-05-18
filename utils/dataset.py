import torch
import random, math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data_path,
        topol_len=128,
        traj_len=200,
        num_grids=256,
        num_roads=8000,
        max_road_length=128,
        max_region_length=64,
        transform=True,
        augment=True,
        dataset_name='chengdu'
    ):
        # config
        self.data_path = data_path
        self.transform = transform
        self.augment = augment
        # length config
        self.topol_len = topol_len
        self.max_road_length = max_road_length
        self.max_region_length = max_region_length
        self.traj_len = traj_len
        # Road segment
        # should be awarded some roads are index from 0
        self.PAD_IDX = 0
        self.BOS_IDX = num_roads - 1
        self.EOS_IDX = num_roads - 2
        self.MASK_IDX = num_roads - 3
        # Region
        self.RAND_IDX = num_grids + 1
        
        # Data augmentation
        if augment:
            self.augment_prob = 0.7
            self.aug_type = ['subsequence', 'reverse', 'random_mask', 'random_drop', 'random_shuffle_local']
            self.p_shuffle = 0.5
            self.p_remove = 0.7
        # normalization
        if dataset_name == 'chengdu':
            self.mean = torch.tensor([104.07596303, 30.68085491], dtype=torch.float32)
            self.std = torch.tensor([2.15106194e-02, 1.89193207e-02], dtype=torch.float32)
        elif dataset_name == 'xian':
            self.mean =  torch.tensor([108.94519514,  34.24489434], dtype=torch.float32)
            self.std = torch.tensor([0.02226231, 0.02007758], dtype=torch.float32)
         # load data
        try:
            with open(self.data_path, "rb") as f:
                self.data = pd.read_pickle(f)
        except:
            raise FileNotFoundError(f"File not found: {self.data_path}")
        
    def __len__(self):
        return len(self.data)
    
    def normalize(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)
        return (tensor - self.mean) / self.std
    
    def denormalize(self, tensor):
        """Denormalize the tensor"""
        return tensor * self.std + self.mean

    def pad_or_truncate(self, tensor, target_len, padding_value=0):
        seq_len = len(tensor)
        if seq_len > target_len:
            if isinstance(tensor, torch.Tensor) and tensor.dim() == 2:  # for trajectory
                tensor = tensor[:target_len]
            else:   
                tensor = torch.cat([tensor[:1], tensor[1:target_len-1], tensor[-1:]])
            attention_mask = torch.ones(target_len)
        else:
            if isinstance(tensor, torch.Tensor) and tensor.dim() == 2:  # for trajectory
                padded_tensor = torch.zeros((target_len, tensor.shape[1]), dtype=tensor.dtype)
            else:  
                padded_tensor = torch.full((target_len,), padding_value, dtype=tensor.dtype)
            attention_mask = torch.zeros(target_len)
            attention_mask[:seq_len] = 1
            padded_tensor[:seq_len] = tensor
            tensor = padded_tensor
        return tensor, attention_mask
    
    # data augmentation method for road
    def augment_road_sequence(self, road_sequence, num_masks=5,drop_prob=0.1,windows_size=3):
        if not self.augment or random.random() >= self.augment_prob:
            return road_sequence
            
        augtype = random.choice(self.aug_type)
        if augtype == 'subsequence':
            if len(road_sequence) <= 3:
                return road_sequence
            start = random.randint(0, len(road_sequence)-2)
            length = random.randint(2, len(road_sequence)-start)
            return road_sequence[start:start+length]
        elif augtype == 'reverse':
            return torch.flip(road_sequence, [0])
        elif augtype == 'random_mask':
            if len(road_sequence) <= num_masks *3:
                return road_sequence
            positions = torch.randperm(len(road_sequence))[:num_masks]
            masked_seq = road_sequence.clone()
            masked_seq[positions] = self.MASK_IDX
            return masked_seq
        elif augtype == 'random_drop':
            if len(road_sequence) <= 3:
                return road_sequence
            mask = torch.rand(len(road_sequence)) > 0.1
            return road_sequence[mask]
        elif augtype == 'random_shuffle_local':
            if len(road_sequence) <= windows_size:
                return road_sequence
            result = road_sequence.clone()
            for i in range(0, len(road_sequence) - windows_size + 1, windows_size):
                window = result[i:i+windows_size]
                perm = torch.randperm(windows_size)
                result[i:i+windows_size] = window[perm]
            return result
        return road_sequence
    
    # data augmentation for region
    def augment_region_sequence(self, region_sequence):
        if not self.augment:
            return region_sequence
            
        prob = random.random()
        if prob < self.p_shuffle:
            window_size = min(5, len(region_sequence)-1)
            start_idx = random.randint(0, len(region_sequence) - window_size)
            window = region_sequence[start_idx:start_idx+window_size].clone()
            perm = torch.randperm(window_size)
            region_sequence[start_idx:start_idx+window_size] = window[perm]
        if prob < self.p_remove:
            keep_mask = torch.rand(len(region_sequence)) > 0.2
            if keep_mask.sum() > 0:
                region_sequence = region_sequence[keep_mask]
        return region_sequence


    
    def __getitem__(self, idx):
        # get data
        traj_df = self.data.iloc[idx]
        # trajectory
        trajectory = torch.tensor(traj_df['trajectory'], dtype=torch.float32)
        original = trajectory[0]
        if self.transform:
            trajectory = self.normalize(trajectory)
        
        # topology
        topology = torch.tensor(traj_df['topology'], dtype=torch.float32)
        if self.transform:
            topology = self.normalize(topology)
        topology, topol_attention_mask = self.pad_or_truncate(topology, self.topol_len)
        
        # road segments
        road_sequence = list(dict.fromkeys(traj_df['roads']))
        road_sequence = torch.tensor(road_sequence, dtype=torch.long)
        if self.augment:
            road_sequence = self.augment_road_sequence(road_sequence)
        road_sequence = torch.cat([
            torch.tensor([self.BOS_IDX]),
            road_sequence,
            torch.tensor([self.EOS_IDX])], dim=0)
        road_sequence, road_attention_mask = self.pad_or_truncate(
            road_sequence, self.max_road_length, padding_value=self.PAD_IDX
        )
        
        # regions
        region_sequence = list(dict.fromkeys(traj_df['cell_sequence']))
        region_sequence = torch.tensor(region_sequence, dtype=torch.long)
        if self.augment:
            region_sequence = self.augment_region_sequence(region_sequence)
        region_sequence, region_attention_mask = self.pad_or_truncate(
            region_sequence, self.max_region_length, padding_value=self.PAD_IDX
        )
        return {
            'label': torch.tensor(idx, dtype=torch.long),
            'trajectory': trajectory,
            'topology': topology,
            'road': road_sequence,
            'region': region_sequence,
            'topology_attention_mask': topol_attention_mask,
            'road_attention_mask': road_attention_mask,
            'region_attention_mask': region_attention_mask,
            'original': original,
        }
        