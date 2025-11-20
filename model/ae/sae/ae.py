"""
Simple Stacked AutoEncoder for time series feature compression
Adapted for KOSPI prediction (time series data)

Input: (batch, seq_len, in_features)
Output: (batch, seq_len, latent_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSAE(nn.Module):
    """
    Simple Stacked AutoEncoder for time series feature compression
    
    Input: (batch, seq_len, in_features)
    Output: (batch, seq_len, latent_dim)
    """
    
    def __init__(self, in_features=9, hidden_dims=[32, 16], latent_dim=8, dropout=0.2, noise_factor=0.05):
        super(SimpleSAE, self).__init__()
        
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.noise_factor = noise_factor
        
        # Encoder
        encoder_layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.extend([
            nn.Linear(prev_dim, latent_dim),
            nn.ReLU()
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, in_features))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to compressed latent space"""
        batch_size, seq_len, features = x.shape
        x = x.view(-1, features)
        z = self.encoder(x)
        z = z.view(batch_size, seq_len, -1)
        return z
    
    def decode(self, z):
        """Decode from latent space to original space"""
        batch_size, seq_len, latent = z.shape
        z = z.view(-1, latent)
        x_recon = self.decoder(z)
        x_recon = x_recon.view(batch_size, seq_len, -1)
        return x_recon
    
    def forward(self, x):
        """Forward pass: Encode and Decode with Denoising"""
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(x) * self.noise_factor
            x_noisy = x + noise
            z = self.encode(x_noisy)
        else:
            z = self.encode(x)
        
        x_recon = self.decode(z)
        return z, x_recon


# Test code
if __name__ == "__main__":
    print("="*70)
    print("SimpleSAE Test for Time Series Data")
    print("="*70)
    
    # Test data: (batch, seq_len, features)
    batch_size = 16
    seq_len = 15
    in_features = 9
    
    x = torch.randn(batch_size, seq_len, in_features)
    print(f"\nInput shape: {x.shape}")
    
    # Create model
    model = SimpleSAE(in_features=9, hidden_dims=[32, 16], latent_dim=8, dropout=0.2)
    
    # Forward pass
    z, x_recon = model(x)
    
    print(f"Encoded shape: {z.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    
    # Check reconstruction error
    recon_error = F.mse_loss(x_recon, x)
    print(f"\nReconstruction Error (MSE): {recon_error.item():.6f}")
    
    print("\n✅ SimpleSAE test completed successfully!")
    print("="*70)
