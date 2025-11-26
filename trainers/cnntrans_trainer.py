"""
CNNTrans Trainer
"""

import torch
import torch.nn as nn
import os
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer
from model.transformer.transformer import CNNTrans


class CNNTransTrainer(BaseTrainer):
    def __init__(self, config, device, config_path='config/config.yaml'):
        super().__init__(config, device, config_path)
        
    def setup_model(self):
        self.model = CNNTrans(
            input_features=self.config['model']['in_features'],
            output_seq_len=self.config['data']['prediction_days'],
            conv_out_channels=self.config['model'].get('conv_out_channels', 252),
            conv_kernel_size=self.config['model'].get('conv_kernel_size', 3),
            d_model=self.config['model'].get('d_model', 512),
            nhead=self.config['model'].get('nhead', 8),
            num_encoder_layers=self.config['model'].get('num_encoder_layers', 3),
            dim_feedforward=self.config['model'].get('dim_feedforward', 2048),
            dropout=self.config['model'].get('dropout', 0.1),
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
    
    def print_training_info(self):
        print(f"\n{'='*70}")
        print(f"Training Configuration")
        print(f"{'='*70}")
        print(f"Model:           {self.config['model']['model_name'].upper()}")
        print(f"Epochs:          {self.config['training']['num_epochs']}")
        print(f"Scheduler:       Warmup({self.config['training']['warmup_epochs']}) + Cosine Annealing")
        print(f"Learning Rate:   {self.config['training']['learning_rate']:.2e} -> {self.config['training']['min_lr']:.2e}")
        print(f"Device:          {self.device}")
        print(f"Checkpoint:      {self.checkpoint_dir}")
        print(f"TensorBoard:     {self.log_dir}")
        print(f"{'='*70}\n")
    
    def train_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        
        src = X
        label = y
        
        output, change_cost = self.model(src)
        
        # last_input = src[:, -1, 0].unsqueeze(-1).unsqueeze(-1)
        # label_ratio = label / (last_input + 1e-8)
        
        loss = self.criterion(output, label)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_norm'])
        self.optimizer.step()
        
        return {
            'total_loss': loss.item()
        }
    
    def validate_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        
        src = X
        label = y
        
        output, change_cost = self.model(src)
        
        # last_input = src[:, -1, 0].unsqueeze(-1).unsqueeze(-1)
        # label_ratio = label / (last_input + 1e-8)
        
        loss = self.criterion(output, label)
        
        return {
            'total_loss': loss.item()
        }
    
    def train_epoch(self, train_loader, epoch):
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} [Train]", 
                         leave=False, dynamic_ncols=True)
        
        batch_count = 0
        for batch_data in train_pbar:
            X, y = batch_data
            
            metrics = self.train_batch(X, y)
            
            train_loss += metrics['total_loss']
            batch_count += 1
            
            train_pbar.set_postfix({
                'loss': f'{train_loss/batch_count:.4f}'
            })
        
        return {
            'total_loss': train_loss / len(train_loader)
        }
    
    def validate_epoch(self, valid_loader, epoch):
        valid_loss = 0.0
        
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} [Valid]", 
                         leave=False, dynamic_ncols=True)
        
        batch_count = 0
        with torch.no_grad():
            for batch_data in valid_pbar:
                X, y = batch_data
                
                metrics = self.validate_batch(X, y)
                
                valid_loss += metrics['total_loss']
                batch_count += 1
                
                valid_pbar.set_postfix({
                    'loss': f'{valid_loss/batch_count:.4f}'
                })
        
        return {
            'total_loss': valid_loss / len(valid_loader)
        }
    
    def print_epoch_summary(self, epoch, train_metrics, valid_metrics, current_lr):
        is_best = valid_metrics['total_loss'] < self.best_valid_loss
        status = "★ NEW BEST" if is_best else f"({self.early_stop_counter+1}/{self.config['training']['patience']})"
        
        print(f"Epoch {epoch+1:3d}/{self.config['training']['num_epochs']} | "
              f"LR: {current_lr:.2e} | "
              f"Train: {train_metrics['total_loss']:.6f} | "
              f"Valid: {valid_metrics['total_loss']:.6f} | "
              f"{status}")
    
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
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pt'))
