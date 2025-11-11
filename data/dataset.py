import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd

class Dataset(Dataset):
    """
    연속적인 시계열 데이터를 슬라이딩 윈도우 방식으로 처리하는 PyTorch Dataset 클래스입니다.
    
    Args:
        data (np.array): 전체 시계열 데이터. shape: (전체 길이, 특징 수)
        input_window (int): 모델에 입력으로 사용할 과거 데이터의 길이 (예: 30일).
        output_window (int): 예측할 미래 데이터의 길이 (예: 5일).
        target_feature_idx (int): 예측 목표가 되는 특징의 인덱스 (예: 종가의 인덱스).
    """
    def __init__(self, data, input_window, output_window, target_feature_idx=3, scaler=None):
        self.input_window = input_window
        self.output_window = output_window
        self.target_feature_idx = target_feature_idx
        row_data = data.copy()
        self.data = data
        
        # 1. Scaler 처리 및 데이터 정규화
        if scaler is None:
            # Scaler가 제공되지 않으면, 학습 데이터셋으로 간주합니다.
            # 새로운 Scaler를 생성하고 fit_transform을 수행합니다.
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(data)

            # scale 확인용 코드
            # i = 0
            # print("Min values of scaled data:\n", self.data[:, i].min())
            # index = np.argmin(self.data[:, i])
            # print("Min values of scaled data index:\n", index)
            # print("Min values of scaled data row:\n", row_data[index, i])

            # print("\nMax values of scaled data:\n", self.data[:, i].max())
            # index = np.argmax(self.data[:, i])
            # print("Max values of scaled data index:\n", index)
            # print("Max values of scaled data row:\n", row_data[index, i])

        else:
            # Scaler가 제공되면, 검증/테스트 데이터셋으로 간주합니다.
            # 제공된 Scaler를 사용하여 transform만 수행합니다.
            self.scaler = scaler
            self.data = self.scaler.transform(data)

        # 데이터로부터 학습 샘플(X, Y) 생성
        self.X, self.Y = self.create_inout_sequences()

    def __len__(self):
        # 전체 샘플의 개수
        return len(self.X)

    def create_inout_sequences(self):
        X_list, Y_list = [], []
        # 전체 데이터 길이에서 입력과 출력 윈도우 길이를 뺀 만큼 반복
        for i in range(len(self.data) - self.input_window - self.output_window + 1):
            # 1. 학습 데이터 (X)
            # i부터 i+input_window까지의 모든 특징을 가져옵니다.
            train_seq = self.data[i : i + self.input_window]
            X_list.append(train_seq)
            
            # 2. 정답 데이터 (Y)
            # 학습 데이터가 끝나는 시점부터 output_window 길이만큼의 '목표 특징'을 가져옵니다.
            target_seq = self.data[i + self.input_window : i + self.input_window + self.output_window, self.target_feature_idx]
            Y_list.append(target_seq)
            
        return np.array(X_list), np.array(Y_list)

    def __getitem__(self, idx):
        # 주어진 인덱스(idx)에 해당하는 학습 데이터와 정답 데이터를 텐서로 변환하여 반환
        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.Y[idx])
        return x, y


# # --- 실행 예시 ---

# # 1. kospi 연속 데이터 생성
# kospi_ticker = '^KS11'
# data = yf.download(kospi_ticker, start='2010-01-01')

# # 필요한 열만 선택하고 결측치 제거
# kospi_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
# kospi_data = kospi_data.dropna()

# # NumPy 배열로 변환
# full_data = kospi_data.values
# print(f"전체 데이터 shape: {full_data.shape}\n")
# print(kospi_data.head())

# # 2. 하이퍼파라미터 설정
# INPUT_WINDOW_SIZE = 30  # 입력으로 사용할 과거 30일치 데이터
# OUTPUT_WINDOW_SIZE = 5   # 예측할 미래 5일치 데이터
# TARGET_FEATURE_INDEX = 3 # 예측 목표는 '종가' (인덱스 3)
# BATCH_SIZE = 64

# # 3. Dataset 인스턴스 생성
# # (여기서는 데이터를 학습/검증/테스트로 나누지 않고 전체를 사용)
# train_dataset = Dataset(
#     data=full_data,
#     input_window=INPUT_WINDOW_SIZE,
#     output_window=OUTPUT_WINDOW_SIZE,
#     target_feature_idx=TARGET_FEATURE_INDEX
# )
# _, y = train_dataset[1]
# print(y[0])
# print(full_data[31, 3])
# # 4. DataLoader 인스턴스 생성
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# print("--- DataLoader 생성 완료 ---")
# print(f"총 샘플 개수: {len(train_dataset)}")
# print(f"배치 크기: {BATCH_SIZE}")
# print(f"총 배치 개수: {len(train_loader)}\n")

# # test_dataset = Dataset(
# #     data=full_data,
# #     input_window=INPUT_WINDOW_SIZE,
# #     output_window=OUTPUT_WINDOW_SIZE,
# #     target_feature_idx=TARGET_FEATURE_INDEX,
# #     scaler=train_dataset.scaler # ★ 중요: 학습용 스케일러를 전달
# # )


# # 5. DataLoader 사용 예시
# # 생성된 DataLoader를 순회하며 데이터 확인
# print("--- DataLoader에서 한 배치(batch) 꺼내기 ---")
# x_batch, y_batch = next(iter(train_loader))

# print(f"학습 데이터(X) 배치 shape: {x_batch.shape}")
# print(f"  -> (배치 크기, 입력 윈도우 길이, 전체 특징 수)")
# print(f"정답 데이터(Y) 배치 shape: {y_batch.shape}")
# print(f"  -> (배치 크기, 출력 윈도우 길이)")
# print("\n첫 번째 샘플 확인:")
# print(f"  X[0] shape: {x_batch[0].shape}")
# print(f"  Y[0] shape: {y_batch[0].shape}")
# print(f"  Y[0] 값: {y_batch[0].numpy()}")

