import torch
import torch.nn as nn


class DirectionalLoss(nn.Module):
    """
    Directional Loss: 방향성(상승/하락)을 맞추는지 평가
    """
    
    def __init__(self):
        super(DirectionalLoss, self).__init__()
    
    def forward(self, predictions, targets, last_close):
        """
        Args:
            predictions: (batch, prediction_days) - 예측 가격
            targets: (batch, prediction_days) - 실제 가격
            last_close: (batch, 1) - 마지막 종가
        """
        pred_direction = torch.sign(predictions - torch.cat([last_close, predictions[:, :-1]], dim=1))
        target_direction = torch.sign(targets - torch.cat([last_close, targets[:, :-1]], dim=1))
        
        direction_error = (pred_direction != target_direction).float()
        
        return direction_error.mean()


class CombinedLoss(nn.Module):
    
    def __init__(self, prediction_weight=50, reconstruction_weight=0.1, directional_weight=0.0):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.directional_loss = DirectionalLoss()
        self.prediction_weight = prediction_weight
        self.reconstruction_weight = reconstruction_weight
        self.directional_weight = directional_weight
    
    def forward(self, predictions, targets, reconstructions, inputs, last_close):
        """
        Args:
            predictions: (batch, prediction_days) - 예측 가격
            targets: (batch, prediction_days) - 실제 목표 가격
            reconstructions: (batch, seq_len, features) - 재구성된 입력
            inputs: (batch, seq_len, features) - 원본 입력
            last_close: (batch, 1) - 마지막 종가
        
        Returns:
            total_loss, pred_loss, recon_loss, dir_loss
        """
        # Prediction loss
        pred_loss = self.mse_loss(predictions, targets)
        
        # Reconstruction loss
        recon_loss = self.mse_loss(reconstructions, inputs)
        
        # Directional loss
        dir_loss = self.directional_loss(predictions, targets, last_close)
        
        # Total loss
        total_loss = (self.prediction_weight * pred_loss + 
                      self.reconstruction_weight * recon_loss + 
                      self.directional_weight * dir_loss)
        
        return total_loss, pred_loss, recon_loss, dir_loss
