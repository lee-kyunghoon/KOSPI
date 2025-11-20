# --- 모델 생성 및 구조 확인 예시 ---
# 모델 인스턴스 생성 (논문 파라미터 기반)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from cnn_trans import CNNTrans
import pandas as pd
import os
import sys
from datetime import datetime
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data.dataset import Dataset

#하이퍼파라미터 설정
batch_size = 32
src_seq_len = 15  # 인코더 입력 시퀀스 길이 (예: 과거 15일)
tgt_seq_len = 5   # 예측 길이 (예: 미래 5일)
input_features = 9 # 입력 데이터의 피처 수 (KOSPI_Close, KOSPI_Open, KOSPI_High, KOSPI_Low, KOSPI_Volume, NASDAQ_Close, US10Y_Yield, VIX_Close, USD_KRW)
output_features = 1
d_model = 512
max_epoch = 100

input_size = 15  # 입력으로 사용할 과거 15일치 데이터
output_size = 5   # 예측할 미래 5일치 데이터

#2. 모델 인스턴스 생성
model = CNNTrans(
    input_features=9,
    conv_out_channels=252,
    conv_kernel_size=3,
    d_model=512,
    nhead=8,
    num_encoder_layers=3,
    dim_feedforward=2048,
    output_size=output_features
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 가능한 장치: {device}")
model.to(device)

# MAPE Loss 함수 정의
def mape_loss(output, target, epsilon=1e-10):
    """
    MAPE (Mean Absolute Percentage Error) 손실 함수
    epsilon: 0으로 나누는 것을 방지하기 위한 작은 값
    """
    return torch.mean(torch.abs((target - output) / (target + epsilon))) * 100

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

model.train()

train_data = pd.read_csv("/home/lgh/kospi/data/train.csv")
train_data = train_data.drop(columns=['Date'])  # 날짜 열 제거
train_data = train_data.values  # pandas DataFrame을 numpy array로 변환

train_dataset = Dataset(
    data=train_data,
    input_window=input_size,
    output_window=output_size,
    target_feature_idx=0,
    batch_shuffle=False,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

valid_data = pd.read_csv("/home/lgh/kospi/data/valid.csv")
valid_data = valid_data.drop(columns=['Date'])  # 날짜 열 제거
valid_data = valid_data.values  # pandas DataFrame을 numpy array로 변환

valid_dataset = Dataset(
    data=valid_data,
    input_window=input_size,
    output_window=output_size,
    target_feature_idx=0,
    scaler=train_dataset.scaler
)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = f"/mnt/d/lgh/kospi/checkpoints/{current_time}"
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"체크포인트 저장 경로: {checkpoint_dir}")

best_valid_loss = float('inf')

def validate(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    
    with torch.no_grad():
        for src, label in valid_loader:
            src = src.to(device)
            label = label.to(device)
            
            label = label.unsqueeze(-1)
            
            output, change_cost = model(src)
            
            last_input = src[:, -1, 0].unsqueeze(-1).unsqueeze(-1)
            label_ratio = label / (last_input + 1e-8)
            
            loss = criterion(output, label) + criterion(change_cost, label_ratio)

            valid_loss += loss.item()
    
    avg_valid_loss = valid_loss / len(valid_loader)
    model.train()
    return avg_valid_loss
    
for epoch in range(max_epoch):
    epoch_loss = 0.0
    for i, (src, label) in enumerate(train_loader):
        src = src.to(device)
        label = label.to(device)
        label = label.unsqueeze(-1)

        optimizer.zero_grad()

        output, change_cost = model(src)

        last_input = src[:, -1, 0].unsqueeze(-1).unsqueeze(-1)
        label_ratio = label / (last_input + 1e-8)
        
        loss = criterion(output, label) + criterion(change_cost, label_ratio)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{max_epoch}], Train Loss: {avg_train_loss:.6f}", end="")

    avg_valid_loss = validate(model, valid_loader, criterion, device)
    print(f", Valid Loss: {avg_valid_loss:.6f}")
        
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        best_model_path = os.path.join(checkpoint_dir, "cnn_trans_best.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
        }, best_model_path)
        print(f"최고 성능 모델 저장됨! Valid Loss: {avg_valid_loss:.6f} → {best_model_path}")
    else:
        print()  # 줄바꿈

final_model_path = os.path.join(checkpoint_dir, "cnn_trans_final.pt")
torch.save({
    'epoch': max_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'best_valid_loss': best_valid_loss,
}, final_model_path)
print(f"\n최종 모델 저장됨: {final_model_path}")
print(f"최고 Validation Loss: {best_valid_loss:.6f}")