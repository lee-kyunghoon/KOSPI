import torch
import torch.nn as nn

from data.RevIN import RevIN

class Conv1dEmbedding(nn.Module):
    """
    1D CNN을 사용하여 입력 데이터의 지역적 특징을 추출하는 임베딩 계층입니다.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        stride = 3
        padding = 0
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
                
    def forward(self, x):
        # 입력 x의 shape: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        return x

class PositionEncoding(nn.Module):
    """
    논문에서 언급한 표 형식의 절대 위치 인코딩입니다.
    각 위치 인덱스에 대해 학습 가능한 임베딩을 사용합니다.
    """
    def __init__(self, d_model, max_len=9):
        super().__init__()
        self.position_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(0)

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        position_encoding = self.position_embed(position_ids)
        position_encoding = position_encoding.permute(1, 0, 2)

        return x + position_encoding

class CNNTrans(nn.Module):
    """
    CNN-Trans-SPP 모델의 전체 구조입니다.
    """
    def __init__(self, input_features=5, conv_out_channels=252, conv_kernel_size=3,
                 d_model=512, nhead=8, num_encoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, output_size=1, max_len=100, output_seq_len=5):
        super().__init__()
        
        self.embedding = Conv1dEmbedding(input_features, conv_out_channels, conv_kernel_size)
        
        self.linear_embed = nn.Linear(conv_out_channels, d_model)

        self.positional_encoding = PositionEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.output_seq_len = output_seq_len
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_seq_len)

        self.linear_out = nn.Linear(d_model, output_size)

        if True:
            self.revin = RevIN(num_features=input_features, affine=True)

    def forward(self, src):
        if True:
            src = self.revin(src, mode='norm')

        last_close = src[:, -1:, 0:1]

        src_embedded = self.embedding(src)
        src_embedded = self.linear_embed(src_embedded)
        
        src_permuted = src_embedded.permute(1, 0, 2)
        
        src_pos = self.positional_encoding(src_permuted)
        
        encoder_output = self.encoder(src_pos)
        encoder_output = encoder_output.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        
        encoder_output = encoder_output.permute(0, 2, 1)
        encoder_output = self.adaptive_pool(encoder_output)
        encoder_output = encoder_output.permute(0, 2, 1)
        
        change_cost = self.linear_out(encoder_output)
        
        output = last_close * change_cost
        if True:
            prediction_expanded = output
            
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
            
            output = prediction_expanded.squeeze(-1)  # (batch, 5)
        

        return output, change_cost



