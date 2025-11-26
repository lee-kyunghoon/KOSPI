import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.ae.model import KOSPIPredictor
from data.dataloader import KospiDataset
import pandas as pd
import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import os
import argparse


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint (automatically detects model type)"""
    # Load checkpoint first to check model type
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model name from checkpoint (fallback to config if not in checkpoint)
    model_name = checkpoint.get('model_name', config['model']['model_name']).lower()
    
    print(f"\n{'='*70}")
    print(f"Loading Model: {model_name.upper()}")
    print(f"{'='*70}")
    
    # Load appropriate model based on model_name
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
        from model.transformer.cnn_trans import CNNTrans
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
        raise ValueError(f"Unknown model name: {model_name}. Supported: 'aecnn', 'cnntrans'")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'valid_loss' in checkpoint:
        print(f"  Valid Loss: {checkpoint['valid_loss']:.6f}")
    print(f"{'='*70}\n")
    
    return model, model_name


def prepare_test_data(config):
    """Prepare test dataset"""
    test_df = pd.read_csv("data/test.csv")
    test_data = test_df.drop(columns=['Date']).values
    
    # Load scaler
    if config['data']['normalization_method'] in ['robust', 'minmax']:
        with open('data/scaler_info.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        test_data = scaler.transform(test_data)
    
    test_dataset = KospiDataset(
        test_data, 
        config['data']['sequence_length'], 
        config['data']['prediction_days'], 
        config['data']['target_feature_idx'],
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False, 
        drop_last=False
    )
    
    return test_loader, scaler


def test_model(model, test_loader, scaler, device, config, model_name):
    """Test model and calculate metrics"""
    criterion = nn.MSELoss()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    print("\n" + "="*70)
    print("Testing Model...")
    print("="*70)
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            # Handle different model outputs
            if model_name == 'cnntrans':
                output, _ = model(X)  # CNNTrans returns (output, change_cost)
                output = output.squeeze(-1)  # (batch, 5, 1) -> (batch, 5)
            else:  # aecnn
                output = model(X, False)  # AECNN needs dates for TimesNet
            
            loss = criterion(output, y)
            
            total_loss += loss.item()
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 5)
    targets = np.concatenate(all_targets, axis=0)  # (N, 5)
    
    # Inverse transform to original scale (Close price only)
    n_features = config['model']['in_features']
    n_pred_days = predictions.shape[1]
    
    predictions_original = np.zeros_like(predictions)
    targets_original = np.zeros_like(targets)
    
    for i in range(n_pred_days):
        # Create dummy array with 9 features
        pred_full = np.concatenate([predictions[:, i:i+1], np.zeros((predictions.shape[0], n_features-1))], axis=1)
        target_full = np.concatenate([targets[:, i:i+1], np.zeros((targets.shape[0], n_features-1))], axis=1)
        
        # Inverse transform
        pred_inv = scaler.inverse_transform(pred_full)
        target_inv = scaler.inverse_transform(target_full)
        
        predictions_original[:, i] = pred_inv[:, 0]
        targets_original[:, i] = target_inv[:, 0]
    
    avg_loss = total_loss / len(test_loader)
    
    # Calculate metrics for each prediction day
    metrics = {}
    for day in range(config['data']['prediction_days']):
        pred_day = predictions_original[:, day]
        target_day = targets_original[:, day]
        
        mse = mean_squared_error(target_day, pred_day)
        mae = mean_absolute_error(target_day, pred_day)
        rmse = np.sqrt(mse)
        mape = calculate_mape(target_day, pred_day)
        r2 = r2_score(target_day, pred_day)
        
        metrics[f'Day_{day+1}'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
    
    # Overall metrics (average across all prediction days)
    all_pred_flat = predictions_original.flatten()
    all_target_flat = targets_original.flatten()
    
    overall_metrics = {
        'MSE': mean_squared_error(all_target_flat, all_pred_flat),
        'RMSE': np.sqrt(mean_squared_error(all_target_flat, all_pred_flat)),
        'MAE': mean_absolute_error(all_target_flat, all_pred_flat),
        'MAPE': calculate_mape(all_target_flat, all_pred_flat),
        'R2': r2_score(all_target_flat, all_pred_flat),
        'Test_Loss': avg_loss
    }
    
    return predictions_original, targets_original, metrics, overall_metrics


def print_last_predictions(predictions, targets, n_samples=5):
    """Print last N predictions"""
    print("\n" + "="*70)
    print(f"Last {n_samples} Test Samples - Detailed Predictions")
    print("="*70)
    
    n_pred_days = predictions.shape[1]
    total_samples = len(predictions)
    
    for sample_idx in range(max(0, total_samples - n_samples), total_samples):
        print(f"\n{'='*70}")
        print(f"Sample {sample_idx + 1} / {total_samples}")
        print(f"{'='*70}")
        print(f"{'Day':<8} {'Actual':<15} {'Predicted':<15} {'Error':<15} {'Error %':<10}")
        print("-"*70)
        
        for day in range(n_pred_days):
            actual = targets[sample_idx, day]
            pred = predictions[sample_idx, day]
            error = pred - actual
            error_pct = (error / actual) * 100 if actual != 0 else 0
            
            print(f"Day {day+1:<4} {actual:<15.2f} {pred:<15.2f} {error:<15.2f} {error_pct:<10.2f}%")
    
    print("\n" + "="*70 + "\n")


def print_metrics(metrics, overall_metrics):
    """Print test metrics"""
    print("\n" + "="*70)
    print("Test Metrics by Prediction Day")
    print("="*70)
    print(f"{'Day':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'R²':<12}")
    print("-"*70)
    
    for day, day_metrics in metrics.items():
        print(f"{day:<10} "
              f"{day_metrics['MSE']:<12.4f} "
              f"{day_metrics['RMSE']:<12.4f} "
              f"{day_metrics['MAE']:<12.4f} "
              f"{day_metrics['MAPE']:<12.2f}% "
              f"{day_metrics['R2']:<12.4f}")
    
    print("\n" + "="*70)
    print("Overall Test Metrics")
    print("="*70)
    print(f"Test Loss (MSE):     {overall_metrics['Test_Loss']:.6f}")
    print(f"MSE:                 {overall_metrics['MSE']:.4f}")
    print(f"RMSE:                {overall_metrics['RMSE']:.4f}")
    print(f"MAE:                 {overall_metrics['MAE']:.4f}")
    print(f"MAPE:                {overall_metrics['MAPE']:.2f}%")
    print(f"R²:                  {overall_metrics['R2']:.4f}")
    print("="*70 + "\n")


def visualize_predictions(predictions, targets, save_dir='results'):
    """Visualize predictions vs targets"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Prediction vs Target for each day
    n_days = predictions.shape[1]
    fig, axes = plt.subplots(1, n_days, figsize=(5*n_days, 4))
    if n_days == 1:
        axes = [axes]
    
    for day in range(n_days):
        pred_day = predictions[:, day]
        target_day = targets[:, day]
        
        axes[day].scatter(target_day, pred_day, alpha=0.5, s=10)
        axes[day].plot([target_day.min(), target_day.max()], 
                      [target_day.min(), target_day.max()], 
                      'r--', lw=2, label='Perfect Prediction')
        axes[day].set_xlabel('Actual', fontsize=12)
        axes[day].set_ylabel('Predicted', fontsize=12)
        axes[day].set_title(f'Day {day+1} Prediction', fontsize=14, fontweight='bold')
        axes[day].legend()
        axes[day].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'), dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'prediction_scatter.png')}")
    plt.close()
    
    # 2. Time series plot - Separate subplot for each prediction day
    n_samples = min(200, len(predictions))
    fig, axes = plt.subplots(n_days, 1, figsize=(15, 4*n_days), sharex=True)
    if n_days == 1:
        axes = [axes]
    
    for day in range(n_days):
        axes[day].plot(range(n_samples), targets[:n_samples, day], 
                      label='Actual', alpha=0.8, linewidth=2, color='blue')
        axes[day].plot(range(n_samples), predictions[:n_samples, day], 
                      label='Predicted', alpha=0.8, linewidth=2, color='red')
        axes[day].fill_between(range(n_samples), 
                              targets[:n_samples, day], 
                              predictions[:n_samples, day], 
                              alpha=0.3, color='gray')
        
        axes[day].set_ylabel('KOSPI Close', fontsize=11)
        axes[day].set_title(f'Day {day+1} Ahead Prediction', fontsize=12, fontweight='bold')
        axes[day].legend(loc='upper left', fontsize=10)
        axes[day].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample Index', fontsize=12)
    fig.suptitle('Time Series Predictions (First 200 Samples)', fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series.png'), dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'time_series.png')}")
    plt.close()
    
    # 2-2. All predictions overlaid (original style for comparison)
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot actual values for Day 1 only (as reference)
    ax.plot(range(n_samples), targets[:n_samples, 0], 
           label='Actual (Day 1)', alpha=0.8, linewidth=2, color='black', linestyle='-')
    
    # Plot predictions for all days
    colors = plt.cm.rainbow(np.linspace(0, 1, n_days))
    for day in range(n_days):
        ax.plot(range(n_samples), predictions[:n_samples, day], 
               label=f'Predicted Day {day+1}', alpha=0.7, linewidth=1.5, color=colors[day])
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('KOSPI Close Price', fontsize=12)
    ax.set_title('All Predictions Overlaid (First 200 Samples)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series_overlay.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(save_dir, 'time_series_overlay.png')}")
    plt.close()
    
    # 3. Error distribution
    fig, axes = plt.subplots(1, n_days, figsize=(5*n_days, 4))
    if n_days == 1:
        axes = [axes]
    
    for day in range(n_days):
        errors = predictions[:, day] - targets[:, day]
        axes[day].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[day].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[day].set_xlabel('Prediction Error', fontsize=12)
        axes[day].set_ylabel('Frequency', fontsize=12)
        axes[day].set_title(f'Day {day+1} Error Distribution\nMean: {errors.mean():.2f}, Std: {errors.std():.2f}', 
                           fontsize=14, fontweight='bold')
        axes[day].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'error_distribution.png')}")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {save_dir}/\n")


def main():
    parser = argparse.ArgumentParser(description='Test KOSPI Prediction Model')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint (e.g., checkpoints/kospi_20251120_123456)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: auto-detect from checkpoint directory)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    args.checkpoint = os.path.join(args.checkpoint, 'best_model.pt')
    args.save_dir = os.path.join("/".join(args.checkpoint.split('/')[:-1]), "result")
    
    # Auto-detect config.yaml from checkpoint directory
    if args.config is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        auto_config_path = os.path.join(checkpoint_dir, 'config.yaml')
        
        if os.path.exists(auto_config_path):
            args.config = auto_config_path
            print(f"\n✓ Auto-detected config: {auto_config_path}")
        else:
            # Fallback to default config
            args.config = 'config/config.yaml'
            print(f"\n⚠ Config not found in checkpoint directory, using default: {args.config}")
    
    # Load config
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("KOSPI Prediction Model - Test Script")
    print("="*70)
    print(f"Device:      {device}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Config:      {args.config}")
    print(f"Save Dir:    {args.save_dir}")
    print("="*70)
    
    # Load model
    model, model_name = load_model(args.checkpoint, config, device)
    
    # Prepare test data
    test_loader, scaler = prepare_test_data(config)
    print(f"\n✓ Test data loaded: {len(test_loader.dataset)} samples")
    
    # Test model
    predictions, targets, metrics, overall_metrics = test_model(
        model, test_loader, scaler, device, config, model_name
    )
    
    # Print last 5 predictions
    print_last_predictions(predictions, targets, n_samples=5)
    
    # Print metrics
    print_metrics(metrics, overall_metrics)
    
    # Visualize results
    visualize_predictions(predictions, targets, args.save_dir)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': list(overall_metrics.keys()),
        'Value': list(overall_metrics.values())
    })
    results_path = os.path.join(args.save_dir, 'test_metrics.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Metrics saved to: {results_path}\n")


if __name__ == "__main__":
    main()
