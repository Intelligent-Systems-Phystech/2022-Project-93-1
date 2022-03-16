import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from typing import Tuple, List, Type, Dict, Any

def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer, 
                       loss_function: torch.nn.Module, 
                       data_loader: torch.utils.data.DataLoader):
    
    model.train()
    train_loss = 0.0
    for data in data_loader:
        x,y = data
    
        optimizer.zero_grad() 
        x = x.to(DEVICE)
  
        y = y.to(DEVICE)
        
        
        y_pred = model(x)
        loss = loss_function(y_pred,y)
              
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return {
        'loss': train_loss / len(data_loader.dataset)
    }

def validate_single_epoch(model: torch.nn.Module,
                          loss_function: torch.nn.Module, 
                          data_loader: torch.utils.data.DataLoader):
    
    model.eval()
    test_loss = 0.0
    
    for data in data_loader:
        x,y = data
        with torch.no_grad():
            x = x.to(DEVICE)
            
            y = y.to(DEVICE)

            y_pred = model(x)
            
            loss = loss_function(y_pred, y)
            test_loss += loss.item()
    
    return {
        'loss': test_loss / len(data_loader.dataset)
    }



def train_model(model: torch.nn.Module, 
                train_dataset: torch.utils.data.Dataset,
                val_dataset: torch.utils.data.Dataset,
                errors,
                loss_function: torch.nn.Module = torch.nn.MSELoss(),
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                optimizer_params: Dict = {},
                lr_scheduler_class: Any = torch.optim.lr_scheduler.StepLR,
                lr_scheduler_params: Dict = {'gamma' : 0.5, 'step_size': 50 },
                batch_size = 16,
                max_epochs = 1000,
                early_stopping_patience = 100,
                initial_lr = 1e-5
):
    optimizer = optimizer_class(model.parameters(), lr=initial_lr, **optimizer_params)
    lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_params)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = None
    best_epoch = None
    
    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1} of {max_epochs}')
        train_metrics = train_single_epoch(model, optimizer, loss_function, train_loader)
        val_metrics = validate_single_epoch(model, loss_function, val_loader)
        errors['test'].append(val_metrics['loss'])
        errors['train'].append(train_metrics['loss'])
        print(f'Validation metrics: \n{val_metrics}')
        
        lr_scheduler.step(val_metrics['loss'])
        
        if best_val_loss is None or best_val_loss > val_metrics['loss'] or epoch % 100 == 0:
            print(f'Best model yet, saving')
            best_val_loss = val_metrics['loss']
           
            best_epoch = epoch
      
        if epoch - best_epoch > early_stopping_patience:
            print('Early stopping triggered')
            return
