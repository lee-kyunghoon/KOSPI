"""
KOSPI Prediction Model
- SimpleSAE: From model/ae/sae/ae.py for time series (1D data)
- Temporal Convolutional Network (TCN) from GitHub: https://github.com/locuslab/TCN/
- TimesNet from GitHub: https://github.com/thuml/Time-Series-Library
- RevIN: Reversible Instance Normalization for better generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model.ae.tcn.tcn import TemporalConvNet
from model.ae.timesnet.TimesNet import Model as TimesNetModel, TimesNetConfigs
from model.ae.sae.ae import SimpleSAE
from data.RevIN import RevIN
from utils.time_features import time_features_from_frequency_str


class KOSPIPredictor(nn.Module):
    """
    KOSPI Prediction Model: SAE + (TCN or TimesNet) + Residual Connection
    Architecture: SAE (9→8) → Backbone → Linear → Residual (5-day forecast)
    """
    
    def __init__(
        self, 
        in_features=9,
        sae_hidden_dims=[32, 16],
        sae_latent_dim=8,
        sae_noise_factor=0.05,
        backbone='tcn',
        tcn_channels=[25, 25, 25, 25],
        tcn_kernel_size=7,
        timesnet_d_model=64,
        timesnet_e_layers=2,
        prediction_days=5,
        sequence_length=15,
        dropout=0.2,
        use_revin=True
    ):
        super(KOSPIPredictor, self).__init__()
        
        self.use_revin = use_revin
        self.prediction_days = prediction_days
        self.sequence_length = sequence_length
        self.backbone_type = backbone
        
        # RevIN for instance normalization
        if self.use_revin:
            self.revin = RevIN(num_features=in_features, affine=True)
        
        self.sae = SimpleSAE(
            in_features=in_features,
            hidden_dims=sae_hidden_dims,
            latent_dim=sae_latent_dim,
            dropout=dropout,
            noise_factor=sae_noise_factor
        )
        
        # Backbone: TCN, TimesNet, or LSTM
        if backbone == 'tcn':
            self.backbone = TemporalConvNet(
                num_inputs=sae_latent_dim,
                num_channels=tcn_channels,
                kernel_size=tcn_kernel_size,
                dropout=dropout
            )
            backbone_output_dim = tcn_channels[-1]
            
        elif backbone == 'timesnet':
            timesnet_configs = TimesNetConfigs(
                seq_len=sequence_length,
                pred_len=prediction_days,
                enc_in=sae_latent_dim,
                d_model=timesnet_d_model,
                e_layers=timesnet_e_layers,
                d_ff=timesnet_d_model * 4,
                top_k=3,
                num_kernels=6,
                dropout=dropout
            )
            self.backbone = TimesNetModel(timesnet_configs)
            self.timesnet_projection = nn.Linear(sae_latent_dim, timesnet_d_model)
            backbone_output_dim = timesnet_d_model
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose 'tcn' or 'timesnet'")
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(backbone_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, prediction_days)
        )
    
    def forward(self, x, return_reconstruction=False, x_dates=None, y_dates=None):
        # RevIN normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        z, x_recon = self.sae(x)
        
        if self.backbone_type == 'tcn':
            z_tcn = z.transpose(1, 2)
            backbone_out = self.backbone(z_tcn)
            features = torch.mean(backbone_out, dim=2)
            
        elif self.backbone_type == 'timesnet':
            
            batch_size, seq_len, _ = z.shape
            device = z.device
            
            # Use actual dates if provided, otherwise zeros
            if x_dates is not None and y_dates is not None:
                # Convert dates to time features
                x_mark_list = []
                x_mark_dec_list = []
                
                for i in range(batch_size):
                    # Convert string dates back to pandas DatetimeIndex
                    x_dates_pd = pd.to_datetime(x_dates[i])
                    y_dates_pd = pd.to_datetime(y_dates[i])
                    
                    x_feat = time_features_from_frequency_str(x_dates_pd, freq='D')
                    y_feat = time_features_from_frequency_str(y_dates_pd, freq='D')
                    x_mark_list.append(x_feat)
                    x_mark_dec_list.append(y_feat)
                
                x_mark = torch.from_numpy(np.array(x_mark_list)).to(device)
                x_mark_dec = torch.from_numpy(np.array(x_mark_dec_list)).to(device)
            else:
                # Fallback to zeros if no dates provided
                x_mark = torch.zeros(batch_size, seq_len, 4, device=device)
                x_mark_dec = torch.zeros(batch_size, self.prediction_days, 4, device=device)
            
            x_dec = torch.zeros(batch_size, self.prediction_days, z.shape[-1], device=device)
            
            backbone_out = self.backbone.forecast(z, x_mark, x_dec, x_mark_dec)
            pooled = torch.mean(backbone_out, dim=1)
            features = self.timesnet_projection(pooled)
        
        delta = self.fc(features)
        
        # 마지막 종가를 5일치 복사하고 각각에 delta 더하기 (누적 오차 방지)
        last_close = x[:, -1, 0].unsqueeze(1)  # (batch, 1)
        last_close_repeated = last_close.repeat(1, self.prediction_days)  # (batch, 5)
        prediction = last_close_repeated + delta
        
        if self.use_revin:
            # (batch, 5) -> (batch, 5, 1)
            prediction_expanded = prediction.unsqueeze(-1)
            
            # RevIN statistics에서 첫 번째 feature만 추출
            # mean: (batch, 1, 9) -> (batch, 1, 1)
            # stdev: (batch, 1, 9) -> (batch, 1, 1)
            mean_close = self.revin.mean[:, :, 0:1]  # (batch, 1, 1)
            stdev_close = self.revin.stdev[:, :, 0:1]  # (batch, 1, 1)
            
            # Denormalize manually
            if self.revin.affine:
                affine_weight_close = self.revin.affine_weight[0]  # scalar
                affine_bias_close = self.revin.affine_bias[0]  # scalar
                prediction_expanded = prediction_expanded - affine_bias_close
                prediction_expanded = prediction_expanded / (affine_weight_close + self.revin.eps * self.revin.eps)
            
            prediction_expanded = prediction_expanded * stdev_close
            prediction_expanded = prediction_expanded + mean_close
            
            prediction = prediction_expanded.squeeze(-1)  # (batch, 5)
        
        if return_reconstruction:
            return prediction, x_recon
        else:
            return prediction
    
    def get_compressed_features(self, x):
        z = self.sae.encode(x)
        return z


def create_kospi_model(backbone='tcn', device='cuda' if torch.cuda.is_available() else 'cpu', dropout=0.2):
    model = KOSPIPredictor(
        in_features=9,
        sae_hidden_dims=[32, 16],
        sae_latent_dim=8,
        backbone=backbone,
        tcn_channels=[25, 25, 25, 25],
        tcn_kernel_size=7,
        prediction_days=5,
        dropout=dropout,
    )
    
    model = model.to(device)
    
    return model



