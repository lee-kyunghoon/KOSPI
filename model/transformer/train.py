# --- 모델 생성 및 구조 확인 예시 ---
# 모델 인스턴스 생성 (논문 파라미터 기반)
import torch
import torch.nn as nn
from cnn_trans import CNNTrans


#1. 하이퍼파라미터 설정
batch_size = 32
src_seq_len = 30  # 인코더 입력 시퀀스 길이 (예: 과거 30일)
tgt_seq_len = 5   # 디코더 입력 및 예측 길이 (예: 미래 5일)
input_features = 5 # 입력 데이터의 피처 수 (예: 시가, 고가, 저가, 종가, 거래량)
output_size = 1   # 예측할 값의 수 (예: 종가)
d_model = 512     # Transformer 모델의 차원

#2. 모델 인스턴스 생성
model = CNNTrans(
    input_features=5,
    conv_out_channels=252,
    conv_kernel_size=3,
    d_model=512,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=2048,
    output_size=output_size
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 가능한 장치: {device}")
model.to(device)

#print(model)

#3. 가상 데이터 생성
src = torch.randn(batch_size, src_seq_len, input_features)
#tgt = torch.randn(batch_size, tgt_seq_len, d_model)
label = torch.randn(batch_size, tgt_seq_len, output_size)

src = src.to(device)
#tgt = tgt.to(device)
label = label.to(device)

#4. 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#5. 모델 학습 과정
model.train()
optimizer.zero_grad()

# 모델 순전파
output = model(src)

# 손실 계산
loss = criterion(output, label)

# 역전파 및 가중치 업데이트
loss.backward()
optimizer.step()

print(f"Input src shape: {src.shape}")
#print(f"Input tgt shape: {tgt.shape}")
print(f"Model output shape: {output.shape}")
print(f"Label shape: {label.shape}")
print(f"Calculated Loss: {loss.item()}")