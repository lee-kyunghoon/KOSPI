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

def get_random_hyperparameters(model_name):
    """
    랜덤 서치를 위한 하이퍼파라미터 공간 정의 및 샘플링
    """

    search_space = {
        'batch_size': [8, 16, 32, 64],
    }

    if(model_name == 'CNNTrans'):
        search_space['d_model'] = [64, 128, 256, 512, 1024]
        search_space['num_encoder_layers'] = [2, 3, 4]
        search_space['nhead'] = [4, 8, 16]
        search_space['dropout'] = [0.1, 0.2, 0.3]
        search_space['dim_feedforward'] = [256, 512, 1024, 2048]
    
    # 랜덤 샘플링
    params = {}
    params['learning_rate'] = random.uniform(1e-5, 1e-3)
    params['batch_size'] = random.choice(search_space['batch_size'])

    if(model_name == 'CNNTrans'):
        params['d_model'] = random.choice(search_space['d_model'])
        params['num_encoder_layers'] = random.choice(search_space['num_encoder_layers'])
        params['nhead'] = random.choice(search_space['nhead'])

    else:
        params['sae_noise_factor'] = random.uniform(0, 0.1)
        params['prediction_weight'] = random.uniform(0, 1)
        params['reconstruction_weight'] = random.uniform(0, 1)
        params['directional_weight'] = random.uniform(0, 1)   

    # d_model은 nhead로 나누어 떨어져야 함
    while params['d_model'] % params['nhead'] != 0:
        params['nhead'] = random.choice(search_space['nhead'])
        
    params['dropout'] = random.choice(search_space['dropout'])
    params['dim_feedforward'] = random.choice(search_space['dim_feedforward'])
    
    return params

def main():
    # 설정
    seq_lengths = [60, 30, 15, 10]
    n_trials_per_seq = 100  # 각 시퀀스 길이별 시도 횟수
    
    config_path = 'config/config.yaml'
    base_config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_loss = float('inf')
    best_params = None
    best_model_path = None
    best_seq_len = None
    
    # 각 시퀀스 길이별 최고 성능 기록용 딕셔너리
    best_per_seq = {}  # {seq_len: {'loss': float, 'path': str, 'params': dict}}
    
    print(f"Starting Random Search for Sequence Lengths: {seq_lengths}")
    print(f"Trials per Sequence Length: {n_trials_per_seq}")
    
    trial_count = 0
    
    for seq_len in seq_lengths:
        print(f"\n{'#'*60}")
        print(f"### Testing Sequence Length: {seq_len}")
        print(f"{'#'*60}")
        
        # 현재 시퀀스 길이의 최고 기록 초기화
        best_per_seq[seq_len] = {'loss': float('inf'), 'path': None, 'params': None}
        
        for i in range(n_trials_per_seq):
            trial_count += 1
            print(f"\n{'='*50}")
            print(f"Trial {trial_count} (SeqLen: {seq_len})")
            print(f"{'='*50}")
            
            # 하이퍼파라미터 샘플링
            params = get_random_hyperparameters(base_config['model']['model_name'])
            print("Sampled Hyperparameters:")
            print(f"  sequence_length: {seq_len}")
            for k, v in params.items():
                print(f"  {k}: {v}")
                
            # 설정 업데이트
            current_config = copy.deepcopy(base_config)
            
            # Data 설정 업데이트 (Sequence Length)
            current_config['data']['sequence_length'] = seq_len
            
            # Training 설정 업데이트
            current_config['training']['learning_rate'] = params['learning_rate']
            current_config['data']['batch_size'] = params['batch_size']
            
            # Model 설정 업데이트 (키가 존재할 때만 업데이트)
            if 'd_model' in params: current_config['model']['d_model'] = params['d_model']
            if 'num_encoder_layers' in params: current_config['model']['num_encoder_layers'] = params['num_encoder_layers']
            if 'nhead' in params: current_config['model']['nhead'] = params['nhead']
            if 'dropout' in params: current_config['model']['dropout'] = params['dropout']
            if 'dim_feedforward' in params: current_config['model']['dim_feedforward'] = params['dim_feedforward']
            
            if 'sae_noise_factor' in params: current_config['model']['sae_noise_factor'] = params['sae_noise_factor']
            if 'prediction_weight' in params: current_config['training']['prediction_weight'] = params['prediction_weight']
            if 'reconstruction_weight' in params: current_config['training']['reconstruction_weight'] = params['reconstruction_weight']
            if 'directional_weight' in params: current_config['training']['directional_weight'] = params['directional_weight']
            
            # 체크포인트 디렉토리 이름 변경 (충돌 방지)
            current_config['checkpoint']['dir'] = f"/mnt/d/lgh/kospi/test/checkpoints/seq{seq_len}_trial_{i+1}"
            
            try:
                # Trainer 초기화 및 학습
                trainer = CNNTransTrainer(current_config, device, config_path)
                
                # 학습 실행
                trainer.train()
                
                val_loss = trainer.best_valid_loss
                print(f"\nTrial {trial_count} Result: Val Loss = {val_loss:.6f}")
                
                # 현재 시퀀스 길이 내에서 최고 성능 갱신
                if val_loss < best_per_seq[seq_len]['loss']:
                    best_per_seq[seq_len]['loss'] = val_loss
                    best_per_seq[seq_len]['path'] = os.path.join(trainer.checkpoint_dir, 'best_model.pt')
                    best_per_seq[seq_len]['params'] = params
                    print(f"★ New Best for SeqLen {seq_len}!")

                # 전체 최고 성능 갱신
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
                    best_seq_len = seq_len
                    best_model_path = os.path.join(trainer.checkpoint_dir, 'best_model.pt')
                    print(f"★ New Overall Best Hyperparameters found!")
                    
            except Exception as e:
                print(f"Trial {trial_count} failed with error: {e}")
                continue
            
    print(f"\n{'='*50}")
    print("Random Search Completed")
    print(f"{'='*50}")
    print(f"Overall Best Validation Loss: {best_loss:.6f}")
    print(f"Overall Best Model Path: {best_model_path}")
    print(f"Overall Best Sequence Length: {best_seq_len}")
    
    # 결과 저장
    with open('best_model_info.txt', 'w', encoding='utf-8') as f:
        # 전체 최고 모델 정보
        f.write("=== Overall Best Model ===\n")
        f.write(f"Validation Loss: {best_loss:.6f}\n")
        f.write(f"Sequence Length: {best_seq_len}\n")
        f.write(f"Model Path: {best_model_path}\n")
        if best_model_path:
            f.write(f"Config Path: {os.path.join(os.path.dirname(best_model_path), 'config.yaml')}\n")
        f.write("Hyperparameters:\n")
        if best_params:
            for k, v in best_params.items():
                f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        # 각 시퀀스 길이별 최고 모델 정보
        f.write("=== Best Model per Sequence Length ===\n")
        for seq_len in seq_lengths:
            info = best_per_seq[seq_len]
            if info['loss'] != float('inf'):
                f.write(f"\n[Sequence Length: {seq_len}]\n")
                f.write(f"Validation Loss: {info['loss']:.6f}\n")
                f.write(f"Model Path: {info['path']}\n")
                if info['path']:
                    f.write(f"Config Path: {os.path.join(os.path.dirname(info['path']), 'config.yaml')}\n")
                f.write("Hyperparameters:\n")
                if info['params']:
                    for k, v in info['params'].items():
                        f.write(f"  {k}: {v}\n")
    
    print(f"\n[Info] Best model info saved to 'best_model_info.txt'")

if __name__ == "__main__":
    main()
