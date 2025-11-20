"""
Time Feature Extraction for TimesNet
"""

import torch
import numpy as np
import pandas as pd


def time_features_from_frequency_str(dates, freq='D'):
    """
    Extract time features from dates
    
    Args:
        dates: pandas DatetimeIndex or array of datetime
        freq: 'D' (daily), 'H' (hourly), etc.
    
    Returns:
        time_features: (len(dates), 4) - [month, day, weekday, hour]
    """
    dates = pd.DatetimeIndex(dates)
    
    # Month: 0~11 normalized to 0~1
    month = (dates.month - 1) / 11.0
    
    # Day: 0~30 normalized to 0~1
    day = (dates.day - 1) / 30.0
    
    # Weekday: 0~6 normalized to 0~1
    weekday = dates.weekday / 6.0
    
    # Hour: 0~23 normalized to 0~1 (for daily data, always 0)
    if freq == 'D':
        hour = np.zeros(len(dates))
    else:
        hour = dates.hour / 23.0
    
    # Stack into (N, 4)
    time_features = np.stack([month, day, weekday, hour], axis=1)
    
    return time_features.astype(np.float32)


def get_time_mark(dates, device='cpu'):
    """
    Get time mark tensor for model input
    
    Args:
        dates: pandas DatetimeIndex or array of datetime
        device: torch device
    
    Returns:
        time_mark: torch.Tensor of shape (len(dates), 4)
    """
    time_feat = time_features_from_frequency_str(dates, freq='D')
    return torch.from_numpy(time_feat).to(device)
