import torch
import torch.nn as nn
from cnn_trans import CNNTrans
# 데이터 로딩 및 전처리를 위한 라이브러리 (예시)
# import numpy as np 
# from sklearn.preprocessing import MinMaxScaler

# --- 1. 모델 클래스 정의 ---
# 이전에 정의한 CNNTransEncoderSPP, Conv1dEmbedding, TabularPositionEncoding 클래스가 
# 이 파일에 있거나 import 되어야 합니다.
# from model_definition import CNNTransEncoderSPP 

# --- 2. 학습된 모델 로드 ---

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"예측에 사용할 장치: {device}")

# 모델 인스턴스 생성 (학습 시와 동일한 하이퍼파라미터 사용)
# 예를 들어, 3일치를 예측하도록 학습했다면 output_size=3
model = CNNTrans(
    input_features=5,
    conv_out_channels=252,
    conv_kernel_size=3,
    d_model=512,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=2048,
    output_size=1
)

# 학습된 가중치 파일(.pth 또는 .pt) 로드
# 'best_model.pth'는 학습 과정에서 저장한 파일명으로 변경해야 합니다.
try:
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    print("학습된 모델 가중치를 성공적으로 로드했습니다.")
except FileNotFoundError:
    print("경고: 저장된 모델 파일을 찾을 수 없습니다. 초기화된 모델로 예측을 수행합니다.")

# 모델을 device로 이동
model.to(device)


# --- 3. 예측을 위한 함수 정의 ---

def predict(model, input_data, device):
    """
    학습된 모델을 사용하여 예측을 수행하는 함수입니다.
    
    Args:
        model (nn.Module): 학습이 완료된 모델.
        input_data (torch.Tensor): 예측에 사용할 입력 데이터. 
                                   shape: (batch_size, seq_len, features)
        device (torch.device): 연산을 수행할 장치 (cpu 또는 cuda).
    
    Returns:
        torch.Tensor: 모델의 예측 결과.
    """
    # 모델을 평가 모드(evaluation mode)로 설정
    # Dropout, BatchNorm 등의 동작을 비활성화하여 일관된 예측 결과를 얻습니다.
    model.eval()
    
    # 기울기 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높입니다.
    with torch.no_grad():
        # 1. 입력 데이터를 모델과 동일한 장치로 이동
        input_data = input_data.to(device)
        
        # 2. 모델을 통해 예측 수행
        prediction = model(input_data)
        
    # 3. 결과를 CPU로 다시 가져와 후처리를 위해 반환
    return prediction.cpu()


# --- 4. 실제 예측 수행 ---

# 예시: 예측에 사용할 새로운 데이터 준비 (과거 5일치 데이터)
# 실제로는 데이터로더나 별도의 전처리 함수를 통해 준비해야 합니다.
# (batch_size=1, seq_len=30, features=5)
new_data_point = torch.randn(1, 30, 5) 

# 정규화(Normalization) 과정이 필요합니다.
# 학습 시 사용했던 Scaler를 그대로 사용해야 합니다.
# scaler = MinMaxScaler()
# new_data_point_normalized = scaler.transform(new_data_point.reshape(-1, 5)).reshape(1, 5, 5)
# new_data_tensor = torch.FloatTensor(new_data_point_normalized)

# 예측 함수 호출
predicted_values = predict(model, new_data_point, device)

# 예측 결과 후처리 (역정규화, Denormalization)
# 학습 시 데이터를 정규화했다면, 예측 결과는 다시 원래의 스케일로 되돌려야 의미를 가집니다.
# predicted_values_original_scale = scaler.inverse_transform(predicted_values)

print("\n--- 예측 결과 ---")
print(f"입력 데이터 shape: {new_data_point.shape}")
print(f"모델 예측 결과 (정규화된 값): {predicted_values}")
print(f"예측 결과 shape: {predicted_values.shape}")
# print(f"모델 예측 결과 (원래 스케일): {predicted_values_original_scale}")

