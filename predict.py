import torch
import torch.nn as nn
from model.ae.model import KOSPIPredictor
import pandas as pd
import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import os
import argparse
from datetime import datetime, timedelta


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_name = checkpoint.get('model_name', config['model']['model_name']).lower()
    
    print(f"\n{'='*70}")
    print(f"모델 로드 중: {model_name.upper()}")
    print(f"{'='*70}")
    
    if model_name == 'aecnn':
        model = KOSPIPredictor(
            in_features=config['model']['in_features'],
            sae_hidden_dims=config['model']['sae_hidden_dims'],
            sae_latent_dim=config['model']['sae_latent_dim'],
            sae_noise_factor=config['model']['sae_noise_factor'],
            backbone=config['model']['backbone'],
            cnn_channels=config['model']['cnn_channels'],
            cnn_kernel_sizes=config['model']['cnn_kernel_size'],
            prediction_days=config['data']['prediction_days'],
            sequence_length=config['data']['sequence_length'],
            dropout=config['model']['dropout'],
            use_revin=config['model']['use_revin']
        ).to(device)
    elif model_name == 'cnntrans':
        from model.transformer.transformer import CNNTrans
        model = CNNTrans(
            input_features=config['model']['in_features'],
            output_seq_len=config['data']['prediction_days'],
            conv_out_channels=config['model'].get('conv_out_channels', 252),
            conv_kernel_size=config['model'].get('conv_kernel_size', 3),
            d_model=config['model'].get('d_model', 512),
            nhead=config['model'].get('nhead', 8),
            num_encoder_layers=config['model'].get('num_encoder_layers', 3),
            dim_feedforward=config['model'].get('dim_feedforward', 2048),
            dropout=config['model'].get('dropout', 0.1),
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 체크포인트 로드: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  에포크: {checkpoint['epoch']}")
    if 'valid_loss' in checkpoint:
        print(f"  검증 손실: {checkpoint['valid_loss']:.6f}")
    print(f"{'='*70}\n")
    
    return model, model_name


def prepare_prediction_data(config, checkpoint_dir, data_path):
    """미래 예측을 위한 데이터 준비"""
    # 데이터 로드
    df = pd.read_csv(data_path)
    dates = df['Date'].values
    data = df.drop(columns=['Date']).values
    
    seq_len = config['data']['sequence_length']
    
    print(f"\n{'='*70}")
    print(f"예측 데이터 준비")
    print(f"{'='*70}")
    print(f"데이터 파일: {data_path}")
    print(f"전체 데이터 개수: {len(data)}개")
    print(f"필요한 시퀀스 길이: {seq_len}개")
    
    if len(data) < seq_len:
        raise ValueError(
            f"❌ 데이터 부족!\n"
            f"   현재 데이터: {len(data)}개\n"
            f"   필요한 데이터: {seq_len}개\n"
            f"   최소 {seq_len}개의 과거 데이터가 필요합니다."
        )
    
    # Scaler 로드
    if config['data']['normalization_method'] in ['robust', 'minmax']:
        scaler_path = os.path.join(checkpoint_dir, 'scaler_info.pkl')
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler 파일을 찾을 수 없습니다: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        data_normalized = scaler.transform(data)
        print(f"✓ Scaler 로드: {scaler_path}")
    else:
        scaler = None
        data_normalized = data
    
    # 가장 최근 sequence_length 개의 데이터를 사용
    input_sequence = data_normalized[-seq_len:]
    input_dates = dates[-seq_len:]
    
    print(f"✓ 입력 시퀀스: {input_dates[0]} ~ {input_dates[-1]}")
    print(f"{'='*70}\n")
    
    return input_sequence, scaler, input_dates, data


def predict_future(model, input_sequence, scaler, config, device, model_name):
    """미래 가격 예측"""
    model.eval()
    
    # 입력 데이터를 tensor로 변환 (batch_size=1 추가)
    X = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)  # (1, seq_len, features)
    
    print(f"\n{'='*70}")
    print(f"예측 중...")
    print(f"{'='*70}")
    print(f"입력 shape: {X.shape}")
    
    with torch.no_grad():
        if model_name == 'cnntrans':
            output, _ = model(X)
            output = output.squeeze(-1)  # (1, pred_days, 1) -> (1, pred_days)
        else:  # aecnn
            output = model(X, False)
    
    predictions = output.cpu().numpy()[0]  # (pred_days,)
    
    print(f"예측 shape: {predictions.shape}")
    print(f"{'='*70}\n")
    
    # 원래 스케일로 복원
    if scaler is not None:
        n_features = config['model']['in_features']
        n_pred_days = len(predictions)
        
        predictions_original = np.zeros(n_pred_days)
        
        for i in range(n_pred_days):
            # Close 가격만 복원 (첫 번째 feature)
            pred_full = np.concatenate([[predictions[i]], np.zeros(n_features-1)])
            pred_inv = scaler.inverse_transform(pred_full.reshape(1, -1))
            predictions_original[i] = pred_inv[0, 0]
    else:
        predictions_original = predictions
    
    return predictions_original


def generate_future_dates(last_date, n_days):
    """미래 날짜 생성 (영업일 기준)"""
    last_date = pd.to_datetime(last_date)
    future_dates = []
    current_date = last_date
    
    while len(future_dates) < n_days:
        current_date += timedelta(days=1)
        # 주말 제외 (월-금만)
        if current_date.weekday() < 5:
            future_dates.append(current_date.strftime('%Y-%m-%d'))
    
    return future_dates


def print_predictions(predictions, future_dates, input_dates, last_price):
    """예측 결과 출력"""
    print(f"\n{'='*70}")
    print(f"KOSPI 종가 예측 결과")
    print(f"{'='*70}")
    print(f"마지막 실제 데이터: {input_dates[-1]} - {last_price:.2f}")
    print(f"{'='*70}")
    print(f"{'날짜':<15} {'예측 가격':<15} {'전일 대비':<15} {'변화율':<10}")
    print("-"*70)
    
    prev_price = last_price
    for i, (date, price) in enumerate(zip(future_dates, predictions)):
        change = price - prev_price
        change_pct = (change / prev_price) * 100
        
        sign = "+" if change >= 0 else ""
        print(f"{date:<15} {price:<15.2f} {sign}{change:<15.2f} {sign}{change_pct:<10.2f}%")
        prev_price = price
    
    print(f"{'='*70}\n")
    
    # 요약 통계
    print(f"{'='*70}")
    print(f"예측 요약")
    print(f"{'='*70}")
    print(f"예측 기간: {future_dates[0]} ~ {future_dates[-1]}")
    print(f"시작 가격: {last_price:.2f}")
    print(f"최종 예측: {predictions[-1]:.2f}")
    print(f"전체 변화: {predictions[-1] - last_price:+.2f} ({(predictions[-1] - last_price) / last_price * 100:+.2f}%)")
    print(f"예측 최고: {predictions.max():.2f}")
    print(f"예측 최저: {predictions.min():.2f}")
    print(f"예측 평균: {predictions.mean():.2f}")
    print(f"{'='*70}\n")


def visualize_predictions(predictions, future_dates, input_dates, historical_data, save_dir='predictions'):
    """예측 결과 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # 과거 데이터 (최근 30일만)
    n_hist = min(30, len(historical_data))
    hist_prices = historical_data[-n_hist:, 0]  # Close price
    hist_dates_display = input_dates[-n_hist:]
    
    # 전체 날짜와 가격
    all_dates = list(hist_dates_display) + future_dates
    all_prices = list(hist_prices) + list(predictions)
    
    # 1. 과거 + 예측 라인 차트
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 과거 데이터
    ax.plot(range(len(hist_prices)), hist_prices, 
           'b-', linewidth=2, label='Actual', marker='o', markersize=4)
    
    # 예측 데이터
    ax.plot(range(len(hist_prices)-1, len(all_prices)), 
           [hist_prices[-1]] + list(predictions),
           'r--', linewidth=2, label='Prediction', marker='s', markersize=6)
    
    # 구분선
    ax.axvline(x=len(hist_prices)-1, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(len(hist_prices)-1, ax.get_ylim()[1]*0.95, '  Current', 
           fontsize=10, color='gray', fontweight='bold')
    
    # X축 날짜 레이블 (일부만 표시)
    step = max(1, len(all_dates) // 10)
    ax.set_xticks(range(0, len(all_dates), step))
    ax.set_xticklabels([all_dates[i] for i in range(0, len(all_dates), step)], 
                       rotation=45, ha='right')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('KOSPI Close Price', fontsize=12, fontweight='bold')
    ax.set_title(f'KOSPI Close Price Prediction ({future_dates[0]} ~ {future_dates[-1]})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_chart.png'), dpi=150, bbox_inches='tight')
    print(f"✓ 저장: {os.path.join(save_dir, 'prediction_chart.png')}")
    plt.close()
    
    # 2. 일별 변화율 바차트
    fig, ax = plt.subplots(figsize=(12, 5))
    
    changes = []
    changes_pct = []
    prev_price = hist_prices[-1]
    
    for price in predictions:
        change = price - prev_price
        change_pct = (change / prev_price) * 100
        changes.append(change)
        changes_pct.append(change_pct)
        prev_price = price
    
    colors = ['red' if x < 0 else 'blue' for x in changes]
    ax.bar(range(len(changes)), changes_pct, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(range(len(future_dates)))
    ax.set_xticklabels(future_dates, rotation=45, ha='right')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily Change (%)', fontsize=12, fontweight='bold')
    ax.set_title('Daily Prediction Changes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'daily_changes.png'), dpi=150, bbox_inches='tight')
    print(f"✓ 저장: {os.path.join(save_dir, 'daily_changes.png')}")
    plt.close()
    
    print(f"\n✓ 모든 시각화 저장 완료: {save_dir}/\n")


def save_predictions_to_csv(predictions, future_dates, save_dir='predictions'):
    """예측 결과를 CSV로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions
    })
    
    csv_path = os.path.join(save_dir, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV 저장: {csv_path}\n")


def main():
    parser = argparse.ArgumentParser(description='KOSPI 미래 가격 예측')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='모델 체크포인트 경로 (예: checkpoints/kospi_20251120_123456)')
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로 (기본값: 체크포인트 디렉토리에서 자동 탐지)')
    parser.add_argument('--data', type=str, default='data/test_final.csv',
                       help='예측에 사용할 데이터 파일 (기본값: data/test_final.csv)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='결과 저장 디렉토리 (기본값: 체크포인트/predictions)')
    
    args = parser.parse_args()
    
    # 체크포인트 경로 설정
    if os.path.isfile(args.checkpoint):
        checkpoint_path = args.checkpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)
    else:
        checkpoint_dir = args.checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    # 저장 디렉토리 설정
    if args.save_dir is None:
        args.save_dir = os.path.join(checkpoint_dir, 'predictions')
    
    # Config 자동 탐지
    if args.config is None:
        auto_config_path = os.path.join(checkpoint_dir, 'config.yaml')
        if os.path.exists(auto_config_path):
            args.config = auto_config_path
            print(f"\n✓ Config 자동 탐지: {auto_config_path}")
        else:
            args.config = 'config/config.yaml'
            print(f"\n⚠ 체크포인트 디렉토리에 config 없음, 기본 config 사용: {args.config}")
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("KOSPI 미래 가격 예측 시스템")
    print("="*70)
    print(f"디바이스:     {device}")
    print(f"체크포인트:   {checkpoint_path}")
    print(f"설정 파일:    {args.config}")
    print(f"데이터 파일:  {args.data}")
    print(f"저장 경로:    {args.save_dir}")
    print("="*70)
    
    # 모델 로드
    model, model_name = load_model(checkpoint_path, config, device)
    
    # 예측 데이터 준비
    input_sequence, scaler, input_dates, historical_data = prepare_prediction_data(
        config, checkpoint_dir, args.data
    )
    
    # 미래 예측
    predictions = predict_future(model, input_sequence, scaler, config, device, model_name)
    
    # 미래 날짜 생성
    future_dates = generate_future_dates(input_dates[-1], config['data']['prediction_days'])
    
    # 마지막 실제 가격
    last_price = historical_data[-1, 0]
    
    # 결과 출력
    print_predictions(predictions, future_dates, input_dates, last_price)
    
    # 시각화
    visualize_predictions(predictions, future_dates, input_dates, historical_data, args.save_dir)
    
    # CSV 저장
    save_predictions_to_csv(predictions, future_dates, args.save_dir)
    
    print("\n" + "="*70)
    print("예측 완료!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
