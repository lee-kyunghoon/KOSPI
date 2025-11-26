import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class KospiDataset(Dataset):
    
    """
    KOSPI 시계열 데이터셋
    
    Args:
        data (np.array): 시계열 데이터. shape: (전체 길이, 특징 수)
        input_window (int): 모델에 입력으로 사용할 과거 데이터의 길이 (default: 30일)
        output_window (int): 예측할 미래 데이터의 길이 (default 5일)
        target_feature_idx (int): 예측 목표가 되는 특징의 인덱스 (default: 0 = 종가)
    """
    def __init__(self, data, input_window, output_window, target_feature_idx=0):
        
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.target_feature_idx = target_feature_idx

        self.X, self.Y = self.create_inout_sequences()

    def __len__(self):
        return len(self.X)

    def create_inout_sequences(self):
        X_list, Y_list = [], []
        
        for i in range(len(self.data) - self.input_window - self.output_window + 1):
            train_seq = self.data[i : i + self.input_window]
            X_list.append(train_seq)
            
            target_seq = self.data[i + self.input_window : i + self.input_window + self.output_window, self.target_feature_idx]
            Y_list.append(target_seq)
        
        return np.array(X_list), np.array(Y_list)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.Y[idx])

        return x, y
    

if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import RobustScaler, MinMaxScaler

    # ========================================
    # 정규화 방법 선택:
    # - 'none': RevIN만 사용 (학습 가능한 정규화)
    # - 'robust': RobustScaler + RevIN (아웃라이어에 강함)
    # - 'minmax': MinMaxScaler + RevIN (0-1 범위 정규화)
    # ========================================
    normalization_method = 'none'  # 'none', 'robust', 'minmax' 중 선택
    
    train_data = pd.read_csv(r"C:\Users\koepm\Desktop\KOSPI\data\test.csv")
    train_data = train_data.drop(columns=['Date']).values
    scaler = None

    if normalization_method == 'none':
        train_data_processed = train_data
        
    elif normalization_method == 'robust':
        scaler = RobustScaler()
        train_data_processed = scaler.fit_transform(train_data)
        
    elif normalization_method == 'minmax':
        scaler = MinMaxScaler()
        train_data_processed = scaler.fit_transform(train_data)
    else:
        raise ValueError("normalization_method는 'none', 'robust', 'minmax' 중 하나여야 합니다.")
    
    # Dataset 생성
    train_dataset = KospiDataset(
        data=train_data_processed,
        input_window=30,
        output_window=5,
        target_feature_idx=0,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for data in train_loader:
        print("=" * 70)
        print("    Input (X) - 30일치 입력 데이터:")
        print(f"   Shape: {data[0].shape}")
        print(f"   Sample (첫 5일): \n{data[0][0, :5, :]}")
        print("\n  Target (Y) - 5일치 예측 목표 (종가):")
        print(f"   Shape: {data[1].shape}")
        print(f"   Values: {data[1][0]}")
        print("=" * 70)

        exit()