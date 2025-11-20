import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class KospiDataset(Dataset):
    
    """
    Args:
        data (np.array): 이미 정규화된 시계열 데이터. shape: (전체 길이, 특징 수)
        input_window (int): 모델에 입력으로 사용할 과거 데이터의 길이 (default: 30일)
        output_window (int): 예측할 미래 데이터의 길이 (default 5일)
        target_feature_idx (int): 예측 목표가 되는 특징의 인덱스
        dates (pd.DatetimeIndex, optional): 날짜 정보
    """
    def __init__(self, data, input_window, output_window, target_feature_idx=0, dates=None):
        
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.target_feature_idx = target_feature_idx
        self.dates = dates

        self.X, self.Y, self.X_dates, self.Y_dates = self.create_inout_sequences()

    def __len__(self):
        return len(self.X)

    def create_inout_sequences(self):
        X_list, Y_list = [], []
        X_dates_list, Y_dates_list = [], []
        
        for i in range(len(self.data) - self.input_window - self.output_window + 1):
            train_seq = self.data[i : i + self.input_window]
            X_list.append(train_seq)
            
            target_seq = self.data[i + self.input_window : i + self.input_window + self.output_window, self.target_feature_idx]
            Y_list.append(target_seq)
            
            if self.dates is not None:
                X_dates_list.append(self.dates[i : i + self.input_window])
                Y_dates_list.append(self.dates[i + self.input_window : i + self.input_window + self.output_window])
            
        X_dates = X_dates_list if self.dates is not None else None
        Y_dates = Y_dates_list if self.dates is not None else None
        
        return np.array(X_list), np.array(Y_list), X_dates, Y_dates

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.Y[idx])
        
        if self.X_dates is not None:
            x_dates = [str(d) for d in self.X_dates[idx]]
            y_dates = [str(d) for d in self.Y_dates[idx]]
            return x, y, x_dates, y_dates
        else:
            return x, y, [], []
    

if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import RobustScaler, MinMaxScaler

    normalization_method = 'robust'
    train_data = pd.read_csv("/home/g1-ubuntu/hi_g1/kospi/data/test2.csv")
    train_data = train_data.drop(columns=['Date']).values

    if normalization_method == 'robust':
        scaler = RobustScaler()
    elif normalization_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Choose 'robust' or 'minmax'")

    train_data_normalized = scaler.fit_transform(train_data)

    train_dataset = KospiDataset(
        data=train_data_normalized,
        input_window=30,
        output_window=5,
        target_feature_idx=0
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for data in train_loader:
        print(f"{data[0]}") # 학습
        print(f"{data[1]}") # 정답
        exit()