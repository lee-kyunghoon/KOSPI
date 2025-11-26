import torch
import torch.nn as nn
import torch.nn.functional as F


class KOSPI_CNN1D(nn.Module):
    def __init__(
        self, 
        latent_dim=8, 
        channels=[16, 32, 64, 128],
        kernel_sizes=[5, 3, 3, 3],
        prediction_days=5,
        dropout=0.2
    ):
        super(KOSPI_CNN1D, self).__init__()
        
        self.latent_dim = latent_dim
        self.prediction_days = prediction_days
        
        self.input_conv = nn.Conv1d(latent_dim, channels[0], kernel_size=1)
        self.input_bn = nn.BatchNorm1d(channels[0])
        self.input_relu = nn.ReLU(inplace=True)
        
        self.blocks = nn.ModuleList()
        in_ch = channels[0]
        
        for i, (out_ch, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            stride = 1 if i < 2 else 2
            
            self.blocks.append(
                CNN1DBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride, dropout=dropout)
            )
            in_ch = out_ch
        
        # Global pooling and attention
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Prediction head
        final_features = channels[-1] * 2
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        
        for block in self.blocks:
            x = block(x)
        
        # pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # (batch, channels[-1])
        max_pool = self.global_max_pool(x).squeeze(-1)  # (batch, channels[-1])
        
        # concat features
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, channels[-1]*2)
        
        return pooled


def create_kospi_cnn(latent_dim=8,
                     prediction_days=5,
                     dropout=0.2,
                     channels=[16, 32, 64, 128],
                     kernel_sizes=[5, 3, 3, 3],):
  
    model = KOSPI_CNN1D(
        latent_dim=latent_dim,
        channels=channels,
        kernel_sizes=kernel_sizes,
        prediction_days=prediction_days,
        dropout=dropout
    )
    return model


class CNN1DBlock(nn.Module):
    """
    Conv → BN → ReLU → Dropout → Conv → BN => 결과
    결과 + 중간 입력 => 최종 출력 (skipped connection)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super(CNN1DBlock, self).__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 
                                     kernel_size=kernel_size, 
                                     stride=stride, 
                                     padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, 
                                     kernel_size=kernel_size, 
                                     stride=1, 
                                     padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 입력과 출력의 shape이 다를 때 조정
        self.use_shortcut_projection = (stride != 1 or in_channels != out_channels)
        
        if self.use_shortcut_projection:
            # 1x1 Conv로 채널 수와 크기를 맞춰줌
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            # 입력을 그대로 통과
            self.shortcut = nn.Identity()
        
        self.final_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual Connection
        identity = self.shortcut(identity)
        out = out + identity
        
        out = self.final_relu(out)
        
        return out

def print_cnn_info(model):
    print("="*70)
    print("KOSPI CNN Model Information")
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Type: {model.__class__.__name__}")
    print("\n" + "="*70)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 16
    seq_len = 30
    latent_dim = 8
    
    x = torch.randn(batch_size, seq_len, latent_dim).to(device)
    model = create_kospi_cnn().to(device)
    print_cnn_info(model)
