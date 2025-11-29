"""
AECNN Trainer
"""

import torch
import torch.nn as nn
import os
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer
from model.ae.model import KOSPIPredictor
from utils.loss import CombinedLoss


class AECNNTrainer(BaseTrainer):
    def __init__(self, config, device, config_path='config/config.yaml'):
        super().__init__(config, device, config_path)
        
    def setup_model(self):
        self.model = KOSPIPredictor(
            in_features=self.config['model']['in_features'],
            sae_hidden_dims=self.config['model']['sae_hidden_dims'],
            sae_latent_dim=self.config['model']['sae_latent_dim'],
            sae_noise_factor=self.config['model']['sae_noise_factor'],
            backbone=self.config['model']['backbone'],
            cnn_channels=self.config['model'].get('cnn_channels', [16, 32, 64, 128]),
            cnn_kernel_sizes=self.config['model'].get('cnn_kernel_size', [5, 3, 3, 3]),
            prediction_days=self.config['data']['prediction_days'],
            sequence_length=self.config['data']['sequence_length'],
            dropout=self.config['model']['dropout'],
            use_revin=self.config['model']['use_revin']
        ).to(self.device)
        
        prediction_weight = self.config['training']['prediction_weight']
        recon_weight = self.config['training']['reconstruction_weight']
        directional_weight = self.config['training'].get('directional_weight', 0.0)
        
        self.criterion = CombinedLoss(
            prediction_weight=prediction_weight,
            reconstruction_weight=recon_weight,
            directional_weight=directional_weight
        )
        
        self.prediction_weight = prediction_weight
        self.recon_weight = recon_weight
        self.directional_weight = directional_weight
    
    def print_training_info(self):
        print(f"\n{'='*70}")
        print(f"Training Configuration")
        print(f"{'='*70}")
        print(f"Model:           {self.config['model']['model_name'].upper()}")
        print(f"Backbone:        {self.config['model']['backbone'].upper()}")
        print(f"Epochs:          {self.config['training']['num_epochs']}")
        print(f"Recon Weight:    {self.recon_weight}")
        print(f"Direction Weight: {self.directional_weight}")
        print(f"Scheduler:       Warmup({self.config['training']['warmup_epochs']}) + Cosine Annealing")
        print(f"Learning Rate:   {self.config['training']['learning_rate']:.2e} -> {self.config['training']['min_lr']:.2e}")
        print(f"Device:          {self.device}")
        print(f"Checkpoint:      {self.checkpoint_dir}")
        print(f"TensorBoard:     {self.log_dir}")
        print(f"{'='*70}\n")
    
    def train_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        
        output, x_recon, x_normalized = self.model(X, return_reconstruction=True)
        last_close = X[:, -1, 0].unsqueeze(1)
        loss, pred_loss, recon_loss, dir_loss = self.criterion(
            predictions=output,
            targets=y,
            reconstructions=x_recon,
            inputs=x_normalized,
            last_close=last_close
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_norm'])
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'pred_loss': pred_loss.item(),
            'recon_loss': recon_loss.item(),
            'dir_loss': dir_loss.item()
        }
    
    def validate_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        
        output, x_recon, x_normalized = self.model(X, return_reconstruction=True)
        
        last_close = X[:, -1, 0].unsqueeze(1)
        loss, pred_loss, recon_loss, dir_loss = self.criterion(
            predictions=output,
            targets=y,
            reconstructions=x_recon,
            inputs=x_normalized,
            last_close=last_close
        )
        
        return {
            'total_loss': loss.item(),
            'pred_loss': pred_loss.item(),
            'recon_loss': recon_loss.item(),
            'dir_loss': dir_loss.item()
        }
    
    def train_epoch(self, train_loader, epoch):
        train_loss = 0.0
        train_pred_loss = 0.0
        train_recon_loss = 0.0
        train_dir_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} [Train]", 
                         leave=False, dynamic_ncols=True)
        
        batch_count = 0
        for batch_data in train_pbar:
            X, y = batch_data
            
            metrics = self.train_batch(X, y)
            
            train_loss += metrics['total_loss']
            train_pred_loss += metrics['pred_loss']
            train_recon_loss += metrics['recon_loss']
            train_dir_loss += metrics['dir_loss']
            batch_count += 1
            
            train_pbar.set_postfix({
                'loss': f'{train_loss/batch_count:.4f}',
                'pred': f'{train_pred_loss/batch_count:.4f}',
                'recon': f'{train_recon_loss/batch_count:.4f}',
                'dir': f'{train_dir_loss/batch_count:.4f}'
            })
        
        return {
            'total_loss': train_loss / len(train_loader),
            'pred_loss': train_pred_loss / len(train_loader),
            'recon_loss': train_recon_loss / len(train_loader),
            'dir_loss': train_dir_loss / len(train_loader)
        }
    
    def validate_epoch(self, valid_loader, epoch):
        valid_loss = 0.0
        valid_pred_loss = 0.0
        valid_recon_loss = 0.0
        valid_dir_loss = 0.0
        
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} [Valid]", 
                         leave=False, dynamic_ncols=True)
        
        batch_count = 0
        with torch.no_grad():
            for batch_data in valid_pbar:
                X, y = batch_data
                
                metrics = self.validate_batch(X, y)
                
                valid_loss += metrics['total_loss']
                valid_pred_loss += metrics['pred_loss']
                valid_recon_loss += metrics['recon_loss']
                valid_dir_loss += metrics['dir_loss']
                batch_count += 1
                
                valid_pbar.set_postfix({
                    'loss': f'{valid_loss/batch_count:.4f}',
                    'pred': f'{valid_pred_loss/batch_count:.4f}',
                    'recon': f'{valid_recon_loss/batch_count:.4f}',
                    'dir': f'{valid_dir_loss/batch_count:.4f}'
                })
        
        return {
            'total_loss': valid_loss / len(valid_loader),
            'pred_loss': valid_pred_loss / len(valid_loader),
            'recon_loss': valid_recon_loss / len(valid_loader),
            'dir_loss': valid_dir_loss / len(valid_loader)
        }
    
    def log_metrics(self, train_metrics, valid_metrics, current_lr, epoch):
        super().log_metrics(train_metrics, valid_metrics, current_lr, epoch)
        
        self.writer.add_scalars('Loss/Prediction', {
            'train': train_metrics['pred_loss'],
            'valid': valid_metrics['pred_loss']
        }, epoch)
        self.writer.add_scalars('Loss/Reconstruction', {
            'train': train_metrics['recon_loss'],
            'valid': valid_metrics['recon_loss']
        }, epoch)
        self.writer.add_scalars('Loss/Directional', {
            'train': train_metrics['dir_loss'],
            'valid': valid_metrics['dir_loss']
        }, epoch)
    
    def print_epoch_summary(self, epoch, train_metrics, valid_metrics, current_lr):
        is_best = valid_metrics['total_loss'] < self.best_valid_loss
        status = "★ NEW BEST" if is_best else f"({self.early_stop_counter+1}/{self.config['training']['patience']})"
        
        print(f"Epoch {epoch+1:3d}/{self.config['training']['num_epochs']} | "
              f"LR: {current_lr:.2e} | "
              f"Train: {train_metrics['total_loss']:.6f} "
              f"(P:{train_metrics['pred_loss']:.6f} R:{train_metrics['recon_loss']:.6f} D:{train_metrics['dir_loss']:.4f}) | "
              f"Valid: {valid_metrics['total_loss']:.6f} "
              f"(P:{valid_metrics['pred_loss']:.6f} R:{valid_metrics['recon_loss']:.6f} D:{valid_metrics['dir_loss']:.4f}) | "
              f"{status}")
    
    def save_checkpoint(self, epoch, avg_valid, additional_metrics=None):
        checkpoint = {
            'epoch': epoch + 1,
            'model_name': self.config['model']['model_name'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'valid_loss': avg_valid,
            'valid_pred_loss': additional_metrics['pred_loss'],
            'valid_recon_loss': additional_metrics['recon_loss'],
            'valid_dir_loss': additional_metrics['dir_loss'],
            'config': self.config,
        }
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pt'))
