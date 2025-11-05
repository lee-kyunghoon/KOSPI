import torch
import torch.nn as nn

class Conv1dEmbedding(nn.Module):
    """
    1D CNN을 사용하여 입력 데이터의 지역적 특징을 추출하는 임베딩 계층입니다.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # padding을 주어 입력 시퀀스 길이가 유지되도록 합니다.
        kernel_size = 6
        stride = 6
        padding = 0
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # 입력 x의 shape: (batch_size, seq_len, features)
        # Conv1d는 (batch_size, features, seq_len) 형태의 입력을 기대합니다.
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        # 다시 (batch_size, seq_len, out_channels) 형태로 변환합니다.
        x = x.permute(0, 2, 1)
        return x

class TabularPositionEncoding(nn.Module):
    """
    논문에서 언급한 표 형식의 절대 위치 인코딩입니다.
    각 위치 인덱스에 대해 학습 가능한 임베딩을 사용합니다.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # 각 위치(0부터 max_len-1)에 대한 임베딩 벡터를 생성합니다.
        self.position_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # 입력 x의 shape: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        # 0부터 seq_len-1까지의 위치 ID를 생성합니다.
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        position_encoding = self.position_embed(position_ids)
        position_encoding = position_encoding.permute(1, 0, 2)
        # 입력 텐서에 위치 임베딩을 더해줍니다.
        return x + position_encoding

class CNNTrans(nn.Module):
    """
    CNN-Trans-SPP 모델의 전체 구조입니다.
    """
    def __init__(self, input_features=5, conv_out_channels=252, conv_kernel_size=3,
                 d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, output_size=1, max_len=500):
        super().__init__()
        
        # 1. Conv1d 임베딩 계층
        self.embedding = Conv1dEmbedding(input_features, conv_out_channels, conv_kernel_size)
        
        # 2. CNN 출력 차원을 Transformer의 d_model 차원과 맞추기 위한 선형 계층
        self.linear_embed = nn.Linear(conv_out_channels, d_model)

        # 3. 위치 인코딩 계층
        self.positional_encoding = TabularPositionEncoding(d_model, max_len)

        # 4. Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 6. 최종 예측을 위한 선형 계층
        self.linear_out = nn.Linear(d_model, output_size)

    def forward(self, src):
        """
        모델의 순방향 전파를 정의합니다.
        - src: 인코더 입력 데이터 (예: 과거 5일의 주가)
        - tgt: 디코더 입력 데이터 (예: 예측 시작점, <SOS> 토큰)
        """
        # src shape: (batch_size, src_seq_len, features)
        # tgt shape: (batch_size, tgt_seq_len, features)

        # 1. Conv1d 임베딩
        src_embedded = self.embedding(src)  # (batch_size, src_seq_len, conv_out_channels)
        
        # 2. 선형 변환으로 d_model 차원 맞추기
        src_embedded = self.linear_embed(src_embedded) # (batch_size, src_seq_len, d_model)
        
        # 3. Transformer 입력 형태로 변환 (seq_len, batch_size, d_model)
        src_permuted = src_embedded.permute(1, 0, 2)
        
        # 4. 위치 인코딩 적용
        src_pos = self.positional_encoding(src_permuted)
        
        # 5. 인코더 통과
        encoder_output = self.encoder(src_pos)

        # # 6. 디코더 입력(tgt)도 동일한 방식으로 처리
        # # 간단한 예시로, tgt도 d_model 차원으로 변환되었다고 가정합니다.
        # # 실제 구현에서는 tgt를 d_model로 매핑하는 과정이 필요합니다.
        # tgt_permuted = tgt.permute(1, 0, 2)
        # tgt_pos = self.positional_encoding(tgt_permuted)

        # # 7. 디코더 통과
        # output = self.decoder(tgt_pos, memory)
        
        # # 8. 최종 출력을 위한 형태 변환 및 선형 계층 통과
        encoder_output = encoder_output.permute(1, 0, 2)  # (batch_size, tgt_seq_len, d_model)
        output = self.linear_out(encoder_output)   # (batch_size, tgt_seq_len, output_size)

        return output



