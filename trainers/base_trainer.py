"""
Base Trainer Class
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import shutil
import yaml
from datetime import datetime
import pickle
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from tqdm import tqdm
from data.dataloader import KospiDataset


class BaseTrainer:
    def __init__(self, config, device, config_path='config/config.yaml'):
        self.config = config
        self.device = device
        self.config_path = config_path
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.best_valid_loss = float('inf')
        self.best_mae_loss = float('inf')
        self.early_stop_counter = 0
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = config['model']['model_name'].lower()
        self.checkpoint_dir = os.path.join(config['checkpoint']['dir'], f"kospi_{timestamp}_{model_name}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        config_dest = os.path.join(self.checkpoint_dir, 'config.yaml')
        # shutil.copy2(self.config_path, config_dest)
        # Save the current config dictionary instead of copying the original file
        with open(config_dest, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        self.log_dir = os.path.join('runs', f"kospi_{timestamp}_{model_name}")
        self.writer = SummaryWriter(self.log_dir)
        
    def prepare_data(self):
        train_df = pd.read_csv("data/train.csv")
        valid_df = pd.read_csv("data/valid.csv")
                
        train_data = train_df.drop(columns=['Date']).values
        valid_data = valid_df.drop(columns=['Date']).values
        
        scaler = None
        if self.config['data']['normalization_method'] in ['robust', 'minmax']:
            scaler = RobustScaler() if self.config['data']['normalization_method'] == 'robust' else MinMaxScaler()
            train_data = scaler.fit_transform(train_data)
            valid_data = scaler.transform(valid_data)
        
        scaler_path = os.path.join(self.checkpoint_dir, 'scaler_info.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to: {scaler_path}")
        
        train_dataset = KospiDataset(
            train_data, 
            self.config['data']['sequence_length'], 
            self.config['data']['prediction_days'], 
            self.config['data']['target_feature_idx'],
        )
        valid_dataset = KospiDataset(
            valid_data, 
            self.config['data']['sequence_length'], 
            self.config['data']['prediction_days'], 
            self.config['data']['target_feature_idx'],
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['data']['batch_size'], 
                                 shuffle=False, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config['data']['batch_size'], 
                                 shuffle=False, drop_last=True)
        
        return train_loader, valid_loader
    
    def setup_optimizer_scheduler(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        warmup_epochs = self.config['training']['warmup_epochs']
        total_epochs = self.config['training']['num_epochs']
        warmup_lr_start = self.config['training']['min_lr']
        max_lr = self.config['training']['learning_rate']
        min_lr = self.config['training']['min_lr']
        
        def get_lr(epoch):
            if epoch < warmup_epochs:
                return warmup_lr_start + (max_lr - warmup_lr_start) * epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=lambda epoch: get_lr(epoch) / max_lr
        )
    
    def setup_model(self):
        raise NotImplementedError("Subclass must implement setup_model")
    
    def train_batch(self, X, y, x_dates, y_dates):
        raise NotImplementedError("Subclass must implement train_batch")
    
    def validate_batch(self, X, y, x_dates, y_dates):
        raise NotImplementedError("Subclass must implement validate_batch")
    
    def print_training_info(self):
        print(f"\n{'='*70}")
        print(f"Training Configuration")
        print(f"{'='*70}")
        print(f"Model:           {self.config['model']['model_name'].upper()}")
        print(f"Epochs:          {self.config['training']['num_epochs']}")
        print(f"Learning Rate:   {self.config['training']['learning_rate']:.2e} -> {self.config['training']['min_lr']:.2e}")
        print(f"Device:          {self.device}")
        print(f"Checkpoint:      {self.checkpoint_dir}")
        print(f"TensorBoard:     {self.log_dir}")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, epoch, avg_valid, additional_metrics=None):
        checkpoint = {
            'epoch': epoch + 1,
            'model_name': self.config['model']['model_name'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'valid_loss': avg_valid,
            'config': self.config,
        }
        
        if additional_metrics:
            checkpoint.update(additional_metrics)
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pt'))
    
    def train(self):
        self.setup_model()
        self.setup_optimizer_scheduler()
        train_loader, valid_loader = self.prepare_data()
        
        self.print_training_info()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            train_metrics = self.train_epoch(train_loader, epoch)
            
            self.model.eval()
            valid_metrics = self.validate_epoch(valid_loader, epoch)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.log_metrics(train_metrics, valid_metrics, current_lr, epoch)
            self.print_epoch_summary(epoch, train_metrics, valid_metrics, current_lr)
            
            avg_valid = valid_metrics['total_loss']
            if avg_valid < self.best_valid_loss:
                self.best_valid_loss = avg_valid
                self.best_mae_loss = valid_metrics['pred_loss']
                self.early_stop_counter = 0
                if self.config['checkpoint']['save_best']:
                    self.save_checkpoint(epoch, avg_valid, valid_metrics)
            else:
                self.early_stop_counter += 1
                
                if self.early_stop_counter >= self.config['training']['patience']:
                    print(f"\n{'='*70}")
                    print("Early stopping triggered!")
                    print(f"{'='*70}")
                    break
        
        if self.config['checkpoint']['save_final']:
            torch.save({
                'model_name': self.config['model']['model_name'],
                'model_state_dict': self.model.state_dict(),
                'best_valid_loss': self.best_valid_loss,
                'config': self.config,
            }, os.path.join(self.checkpoint_dir, 'final_model.pt'))
        
        self.writer.close()
        
        print(f"\n{'='*70}")
        print(f"Training Completed!")
        print(f"{'='*70}")
        print(f"Best Valid Loss:     {self.best_valid_loss:.6f}")
        print(f"Checkpoint Dir:      {self.checkpoint_dir}")
        print(f"TensorBoard Logs:    {self.log_dir}")
        print(f"{'='*70}")
        print(f"\nTo view TensorBoard:")
        print(f"  tensorboard --logdir=runs")
        print(f"{'='*70}\n")
    
    def train_epoch(self, train_loader, epoch):
        raise NotImplementedError("Subclass must implement train_epoch")
    
    def validate_epoch(self, valid_loader, epoch):
        raise NotImplementedError("Subclass must implement validate_epoch")
    
    def log_metrics(self, train_metrics, valid_metrics, current_lr, epoch):
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        self.writer.add_scalars('Loss/Total', {
            'train': train_metrics['total_loss'],
            'valid': valid_metrics['total_loss']
        }, epoch)
    
    def print_epoch_summary(self, epoch, train_metrics, valid_metrics, current_lr):
        is_best = valid_metrics['total_loss'] < self.best_valid_loss
        status = "★ NEW BEST" if is_best else f"({self.early_stop_counter+1}/{self.config['training']['patience']})"
        
        print(f"Epoch {epoch+1:3d}/{self.config['training']['num_epochs']} | "
              f"LR: {current_lr:.2e} | "
              f"Train: {train_metrics['total_loss']:.6f} | "
              f"Valid: {valid_metrics['total_loss']:.6f} | "
              f"{status}")

