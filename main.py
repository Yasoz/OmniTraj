import os, shutil,datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.utils as utils
from utils.dataset import TrajectoryDataset
from utils.omni_semantic import OmniModel
from utils.utils import AvgMeter, get_lr
from utils.config import encoder_configs,dataset_configs
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d

def setup_experiment_directories(Exp_name="Omni",dataset_name="CD"):
    root_dir = Path(__file__).resolve().parent
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    experiment_dir = root_dir / Exp_name / f"{dataset_name}-{timestamp}"
    files_save = experiment_dir / "Files"
    model_save = experiment_dir / "models"
    for directory in [files_save, model_save]:
        directory.mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(root_dir / "utils"):
        if filename.endswith(".py"):
            shutil.copy(root_dir / "utils" / filename, files_save)
    this_file = Path(__file__)
    shutil.copy(this_file, files_save)
    print("All files saved path ---->>", experiment_dir)
    return  model_save

def build_loaders(file_path, config, mode = "train"):
    dataset_params = config["dataset_params"]
    loader_params = config["loader_params"][mode]
    dataset = TrajectoryDataset( data_path=file_path,  **dataset_params )
    dataloader = DataLoader(dataset,  **loader_params)
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader),disable=True)
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items()}
        fusion_modality = ['topology_road', 'topology_region', 'road_region', 'topology_road_region']
        loss = model(batch, fusion_modality)
        if loss.dim() > 0:  # if loss is not scalar
            loss = loss.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf in loss, abort/skip this batch.")
        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #add gradient clipping
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        
        count = batch["trajectory"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader),disable=True)
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items()}
        fusion_modality = ['topology_road', 'topology_region', 'road_region', 'topology_road_region']
        loss = model(batch,fusion_modality)
        if loss.dim() > 0:  # if loss is not scalar
            loss = loss.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf in loss, abort/skip this batch.")
        count = batch["trajectory"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main(config):
    num_epochs = 500
    train_loader = build_loaders(config['train_file_path'], config, mode="train")
    valid_loader = build_loaders(config['test_file_path'], config, mode="valid")
    
    model_config = dict_to_namespace(encoder_configs)
    contrast_pairs = [('trajectory', 'topology'), ('topology', 'road'), ('topology', 'region')]
    fusion_modality = ['topology_road', 'topology_region', 'road_region', 'topology_road_region']
    model = OmniModel(model_config, contrast_pairs, projection_dim=512)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"Model Parameters: {num_params:.2f} M")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,           
        betas=(0.9, 0.999),
        weight_decay=1e-4 
    )
    #  Learning rate scheduler：Cosine Annealing
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,   
        eta_min=1e-5       
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"[------Epoch: {epoch + 1} -------]")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        time = pd.Timestamp.now().strftime("%H:%M:%S")
        print(f"{time}: Train Loss: {train_loss}")
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            time = pd.Timestamp.now().strftime("%H:%M:%S")
            print(f"{time}: Valid Loss: {valid_loss}")
            
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            m_path = config['model_save'] + "/best_OmniModel.pt"
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), m_path)
            else:
                torch.save(model.state_dict(), m_path)
            print("Saved Best Model at epoch {} with loss {}!".format(epoch, best_loss))
            
        if epoch % 20 == 0:
            m_path = config['model_save'] + f"/epoch_{epoch}.pt"
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), m_path)
            else:
                torch.save(model.state_dict(), m_path)
            print("Saved Epoch Model!")
        lr_scheduler.step()

if __name__ == "__main__":
    model_save = setup_experiment_directories(Exp_name="OmniModel",dataset_name="XA")
    dataset_name ='xian'
    path = {
        "train_file_path": f'../Trajdata/{dataset_name}_train.pkl',
        "test_file_path": f'../Trajdata/{dataset_name}_val.pkl'
    }
    config = dataset_configs[dataset_name]
    config.update(path)
    config['model_save'] = str(model_save)
    
    main(dataset_configs[dataset_name])
    