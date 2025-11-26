import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model.ae.sae.ae import SimpleSAE
from model.ae.cnn.cnn_layer import create_kospi_cnn
from data.RevIN import RevIN

class KOSPIPredictor(nn.Module):
    
    def __init__(
        self, 
        in_features=9,
        sae_hidden_dims=[32, 16],
        sae_latent_dim=8,
        sae_noise_factor=0.05,
        backbone='cnn1d',
        cnn_channels=[16, 32, 64, 128],
        cnn_kernel_sizes=[5, 3, 3, 3],
        prediction_days=5,
        sequence_length=30,
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
        
        if backbone == 'cnn1d':
            self.backbone = create_kospi_cnn(
                latent_dim=sae_latent_dim,
                prediction_days=prediction_days,
                dropout=dropout,
                kernel_sizes=cnn_kernel_sizes,
                channels=cnn_channels
            )

            backbone_output_dim = cnn_channels[-1] * 2
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
    
    def forward(self, x, return_reconstruction=False):
        # RevIN normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        z, x_recon = self.sae(x)
        
        if self.backbone_type == 'cnn1d':
            features = self.backbone(z)
        
        delta = self.fc(features)
        
        # 마지막 종가를 5일치 복사하고 각각에 delta 더하기
        last_close = x[:, -1, 0].unsqueeze(1)  # (batch, 1)
        last_close_repeated = last_close.repeat(1, self.prediction_days)  # (batch, 5)
        prediction = last_close_repeated + delta
        
        if self.use_revin:
            prediction_expanded = prediction.unsqueeze(-1)
            
            # RevIN statistics에서 첫 번째 feature만 추출
            mean_close = self.revin.mean[:, :, 0:1]
            stdev_close = self.revin.stdev[:, :, 0:1]
            
            # Denormalize manually
            if self.revin.affine:
                affine_weight_close = self.revin.affine_weight[0]  # scalar
                affine_bias_close = self.revin.affine_bias[0]  # scalar
                prediction_expanded = prediction_expanded - affine_bias_close
                prediction_expanded = prediction_expanded / (affine_weight_close + self.revin.eps * self.revin.eps)
            
            prediction_expanded = prediction_expanded * stdev_close
            prediction_expanded = prediction_expanded + mean_close
            
            prediction = prediction_expanded.squeeze(-1)
        
        if return_reconstruction:
            return prediction, x_recon
        else:
            return prediction
    
    def get_compressed_features(self, x):
        z = self.sae.encode(x)
        return z


def create_kospi_model(backbone='cnn1d', device='cuda' if torch.cuda.is_available() else 'cpu', dropout=0.2):
    model = KOSPIPredictor(
        in_features=9,
        sae_hidden_dims=[32, 16],
        sae_latent_dim=8,
        backbone=backbone,
        prediction_days=5,
        dropout=dropout,
    )
    
    model = model.to(device)
    
    return model



