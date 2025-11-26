import torch
import yaml
import random
import copy
import os
import sys
from trainers.cnntrans_trainer import CNNTransTrainer

# 랜덤 시드 설정
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_random_hyperparameters():
    """
    랜덤 서치를 위한 하이퍼파라미터 공간 정의 및 샘플링
    """
    # 검색 공간 정의
    search_space = {
        'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        'batch_size': [8, 16, 32, 64],
        'd_model': [128, 256, 512],
        'num_encoder_layers': [2, 3, 4],
        'nhead': [4, 8],
        'dropout': [0.1, 0.2, 0.3],
        'dim_feedforward': [512, 1024, 2048]
    }
    
    # 랜덤 샘플링
    params = {}
    params['learning_rate'] = random.choice(search_space['learning_rate'])
    params['batch_size'] = random.choice(search_space['batch_size'])
    params['d_model'] = random.choice(search_space['d_model'])
    params['num_encoder_layers'] = random.choice(search_space['num_encoder_layers'])
    params['nhead'] = random.choice(search_space['nhead'])
    
    # d_model은 nhead로 나누어 떨어져야 함
    while params['d_model'] % params['nhead'] != 0:
        params['nhead'] = random.choice(search_space['nhead'])
        
    params['dropout'] = random.choice(search_space['dropout'])
    params['dim_feedforward'] = random.choice(search_space['dim_feedforward'])
    
    return params

def main():
    # 설정
    n_trials = 10  # 시도할 횟수
    config_path = 'config/config.yaml'
    base_config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_loss = float('inf')
    best_params = None
    best_model_path = None
    
    print(f"Starting Random Search for {n_trials} trials...")
    
    for i in range(n_trials):
        print(f"\n{'='*50}")
        print(f"Trial {i+1}/{n_trials}")
        print(f"{'='*50}")
        
        # 하이퍼파라미터 샘플링
        params = get_random_hyperparameters()
        print("Sampled Hyperparameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
            
        # 설정 업데이트
        current_config = copy.deepcopy(base_config)
        
        # Training 설정 업데이트
        current_config['training']['learning_rate'] = params['learning_rate']
        current_config['data']['batch_size'] = params['batch_size']
        
        # Model 설정 업데이트
        current_config['model']['d_model'] = params['d_model']
        current_config['model']['num_encoder_layers'] = params['num_encoder_layers']
        current_config['model']['nhead'] = params['nhead']
        current_config['model']['dropout'] = params['dropout']
        current_config['model']['dim_feedforward'] = params['dim_feedforward']
        
        # 체크포인트 디렉토리 이름 변경 (충돌 방지)
        current_config['checkpoint']['dir'] = f"/mnt/d/lgh/kospi/test/checkpoints/trial_{i+1}"
        
        try:
            # Trainer 초기화 및 학습
            trainer = CNNTransTrainer(current_config, device, config_path)
            
            # 학습 실행 (여기서는 전체 에포크를 다 돌지만, 필요하면 에포크 수를 줄여서 테스트 가능)
            # 예: current_config['training']['num_epochs'] = 10 
            trainer.train()
            
            val_loss = trainer.best_valid_loss
            print(f"\nTrial {i+1} Result: Val Loss = {val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
                # best_model.pt가 저장된 실제 경로 저장
                best_model_path = os.path.join(trainer.checkpoint_dir, 'best_model.pt')
                print(f"★ New Best Hyperparameters found!")
                
        except Exception as e:
            print(f"Trial {i+1} failed with error: {e}")
            continue
            
    print(f"\n{'='*50}")
    print("Random Search Completed")
    print(f"{'='*50}")
    print(f"Best Validation Loss: {best_loss:.6f}")
    print(f"Best Model Path: {best_model_path}")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
