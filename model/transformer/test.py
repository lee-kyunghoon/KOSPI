import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn_trans import CNNTrans
import pandas as pd
import os
import numpy as np
import sys

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data.dataset import Dataset

#1. 하이퍼파라미터 설정 (학습 시와 동일하게)
batch_size = 1
INPUT_WINDOW_SIZE = 15  # 입력으로 사용할 과거 15일치 데이터 (학습 시 설정과 동일해야 함)
OUTPUT_WINDOW_SIZE = 5   # 예측할 미래 5일치 데이터
TARGET_FEATURE_INDEX = 0 # 예측 목표는 'KOSPI_Close' (인덱스 0)
input_features = 9
output_size = 1
d_model = 512

#2. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 가능한 장치: {device}")

#3. 모델 인스턴스 생성
model = CNNTrans(
    input_features=9,
    conv_out_channels=252,
    conv_kernel_size=3,
    d_model=512,
    nhead=8,
    num_encoder_layers=3,
    dim_feedforward=2048,
    output_size=output_size
)
model.to(device)

#4. 저장된 모델 불러오기
checkpoint_path = "/mnt/d/lgh/kospi/checkpoints/20251120_125134/cnn_trans_best.pt"  # 원하는 체크포인트 경로
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"모델 로드 완료: {checkpoint_path}")
print(f"로드된 에포크: {checkpoint['epoch']}, 학습 손실: {checkpoint['train_loss']:.6f}")
#5. Train 데이터 로드 (scaler를 얻기 위해)
train_data = pd.read_csv("/home/lgh/kospi/data/train.csv")
train_data = train_data.drop(columns=['Date'])
train_data = train_data.values

train_dataset = Dataset(
    data=train_data,
    input_window=INPUT_WINDOW_SIZE,
    output_window=OUTPUT_WINDOW_SIZE,
    target_feature_idx=TARGET_FEATURE_INDEX
)

test_data = pd.read_csv("/home/lgh/kospi/data/test2.csv")
test_dates = test_data['Date'].values
test_data = test_data.drop(columns=['Date'])
test_data_raw = test_data.values

test_dataset = Dataset(
    data=test_data_raw,
    input_window=INPUT_WINDOW_SIZE,
    output_window=OUTPUT_WINDOW_SIZE,
    target_feature_idx=TARGET_FEATURE_INDEX,
    scaler=train_dataset.scaler
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

model.eval()

print("\n=== 테스트 데이터 예측 시작 ===\n")

# 역정규화를 위한 준비
scaler = test_dataset.scaler
kospi_close_idx = TARGET_FEATURE_INDEX
min_val = scaler.data_min_[kospi_close_idx]
scale_val = scaler.scale_[kospi_close_idx]

all_actuals = []
all_predictions = []

print("=" * 100)
print(f"{'배치':<8} {'일차':<6} {'실제 KOSPI 종가':<25} {'예측 KOSPI 종가':<25} {'차이(원)':<15}")
print("=" * 100)

with torch.no_grad():
    for batch_idx, (src, label) in enumerate(test_loader):
        src = src.to(device)
        label = label.to(device)
        
        output,_ = model(src)
        
        output_np = output.cpu().numpy()
        label_np = label.cpu().numpy()
        
        # 배치 내의 각 샘플에 대해 처리 (batch_size=1이므로 한 번만 반복)
        for sample_idx in range(output_np.shape[0]):
            pred_sample = output_np[sample_idx].squeeze()
            actual_sample = label_np[sample_idx]
            
            pred_denorm = pred_sample / scale_val + min_val
            actual_denorm = actual_sample / scale_val + min_val
            
            all_actuals.extend(actual_sample)
            all_predictions.extend(pred_sample)
            
            for day in range(OUTPUT_WINDOW_SIZE):
                actual_price = actual_denorm[day]
                pred_price = pred_denorm[day]
                diff = abs(actual_price - pred_price)
                
                print(f"{batch_idx:<8} {day+1:<6} {actual_price:<25.2f} {pred_price:<25.2f} {diff:<15.2f}")
            print("-" * 100)

all_actuals = np.array(all_actuals)
all_predictions = np.array(all_predictions)

mae = np.mean(np.abs(all_actuals - all_predictions))
mape = np.mean(np.abs((all_actuals - all_predictions)/all_actuals))
mse = np.mean((all_actuals - all_predictions) ** 2)
rmse = np.sqrt(mse)

print("\n=== 전체 예측 성능 (정규화된 값 기준) ===")
print(f"MAE (Mean Absolute Error): {mae:.6f}")
print(f"MSE (Mean Squared Error): {mse:.6f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.6f}")

actuals_denorm = all_actuals / scale_val + min_val
predictions_denorm = all_predictions / scale_val + min_val

print("\n=== 전체 예측 성능 (실제 KOSPI 종가 기준) ===")
mae_denorm = np.mean(np.abs(actuals_denorm - predictions_denorm))
mse_denorm = np.mean((actuals_denorm - predictions_denorm) ** 2)
mape_denorm = np.mean(np.abs((actuals_denorm - predictions_denorm)/actuals_denorm)) * 100

print(f"총 배치 수: {batch_idx + 1}")
print(f"총 예측 개수: {len(all_predictions)}")
print(f"MAE: {mae_denorm:.2f} 원")
print(f"MSE: {mse_denorm:.2f}")
print(f"MAPE: {mape_denorm:.6f}")

print("\n테스트 완료!")
