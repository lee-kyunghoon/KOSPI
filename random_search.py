import torch
import yaml
import random
import copy
import os
import json
from datetime import datetime
from trainers.cnntrans_trainer import CNNTransTrainer
from trainers.aecnn_trainer import AECNNTrainer

def main():
    n_trials = 100 # 시도할 하이퍼파라미터 조합 수
    top_k = 10  # 상위 k개 저장
    config_path = 'config/config.yaml'
    base_config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_loss = float('inf')
    best_params = None
    best_model_path = None
    
    # 상위 k개 결과 저장용 리스트
    top_results = []
    
    print(f"Starting Random Search for {n_trials} trials...")
    print(f"Top {top_k} results will be saved.")
    
    for i in range(n_trials):
        print(f"\n{'='*50}")
        print(f"Trial {i+1}/{n_trials}")
        print(f"{'='*50}")
        
        # 하이퍼파라미터 샘플링
        params = get_random_hyperparameters(base_config['model']['model_name'])
        print("Sampled Hyperparameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
            
        # 설정 업데이트
        current_config = copy.deepcopy(base_config)
        
        # Training 설정 업데이트
        current_config['training']['learning_rate'] = params['learning_rate']
        current_config['data']['batch_size'] = params['batch_size']
        current_config['data']['sequence_length'] = params['sequence_length']
        current_config['model']['dropout'] = params['dropout']
        current_config['data']['normalization_method'] = params['normalization_method']
        
        # Model 설정 업데이트
        if base_config['model']['model_name'].lower() == "cnntrans":
            current_config['model']['d_model'] = params['d_model']
            current_config['model']['num_encoder_layers'] = params['num_encoder_layers']
            current_config['model']['nhead'] = params['nhead']
            current_config['model']['dim_feedforward'] = params['dim_feedforward']
        else:
            current_config['model']['sae_noise_factor'] = params['sae_noise_factor']
            current_config['training']['prediction_weight'] = params['prediction_weight']
            current_config['training']['reconstruction_weight'] = params['reconstruction_weight']
            current_config['training']['directional_weight'] = params['directional_weight'] 

        # 체크포인트 디렉토리 이름 변경
        current_config['checkpoint']['dir'] = f"/mnt/d/lgh/kospi/test/checkpoints/trial_{i+1}"
        
        try:
            # Trainer 초기화 및 학습
            if base_config['model']['model_name'].lower() == "cnntrans":
                trainer = CNNTransTrainer(current_config, device, config_path)
            else:
                trainer = AECNNTrainer(current_config, device, config_path)

            trainer.train()
            
            if base_config['model']['model_name'].lower() == "cnntrans":
                val_loss = trainer.best_valid_loss
            else:
                val_loss = trainer.best_mae_loss

            print(f"\nTrial {i+1} Result: Val Loss = {val_loss:.6f}")
            
            # 상위 k개 업데이트
            result = {
                'trial': i + 1,
                'val_loss': val_loss,
                'params': params,
                'model_path': os.path.join(trainer.checkpoint_dir, 'best_model.pt')
            }
            top_results.append(result)
            top_results.sort(key=lambda x: x['val_loss'])
            top_results = top_results[:top_k]  # 상위 k개만 유지
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
                best_model_path = os.path.join(trainer.checkpoint_dir, 'best_model.pt')
                print(f"★ New Best Hyperparameters found!")
                
        except Exception as e:
            print(f"Trial {i+1} failed with error: {e}")
            continue
    
    # 상위 k개 결과 저장
    save_top_results(top_results, base_config['model']['model_name'], top_k)
            
    print(f"\n{'='*50}")
    print("Random Search Completed")
    print(f"{'='*50}")
    print(f"Best Validation Loss: {best_loss:.6f}")
    print(f"Best Model Path: {best_model_path}")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    print(f"\n{'='*50}")
    print(f"Top {top_k} Results Summary")
    print(f"{'='*50}")
    for idx, result in enumerate(top_results, 1):
        print(f"\nRank {idx}: Trial {result['trial']}")
        print(f"  Val Loss: {result['val_loss']:.6f}")
        print(f"  Model: {result['model_path']}")
    
    print(f"\nTop {top_k} results saved to: test/random_search_top{top_k}_results.json")

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_top_results(top_results, model_name, top_k):
    os.makedirs('test', exist_ok=True)
    
    # 타임스탬프 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 저장할 데이터 구성
    save_data = {
        'metadata': {
            'model_name': model_name,
            'top_k': top_k,
            'timestamp': timestamp,
            'total_trials': len(top_results)
        },
        'results': []
    }
    
    for idx, result in enumerate(top_results, 1):
        save_data['results'].append({
            'rank': idx,
            'trial': result['trial'],
            'val_loss': float(result['val_loss']),
            'model_path': result['model_path'],
            'hyperparameters': result['params']
        })
    
    output_path = f'test/random_search_top{top_k}_results_{timestamp}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    latest_path = f'test/random_search_top{top_k}_results_latest.json'
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved:")
    print(f"   - {output_path}")
    print(f"   - {latest_path}")
    
    txt_path = f'test/random_search_top{top_k}_results_{timestamp}.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"Random Search Top {top_k} Results\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        for idx, result in enumerate(top_results, 1):
            f.write(f"Rank {idx}: Trial {result['trial']}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Validation Loss: {result['val_loss']:.6f}\n")
            f.write(f"Model Path: {result['model_path']}\n")
            f.write(f"\nHyperparameters:\n")
            for k, v in result['params'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n\n")
    
    print(f"   - {txt_path}")


def get_random_hyperparameters(model_name):
    """
    랜덤 서치를 위한 하이퍼파라미터 샘플링
    """

    search_space = {
        'batch_size': [8, 16, 32, 64],
        'dropout': [0.1, 0.2, 0.3],
        'sequence_length': [15, 30, 60, 120],
        'normalization_method': ['robust', 'minmax', 'None']
    }

    if(model_name.lower() == 'cnntrans'):
        search_space['d_model'] = [128, 256, 512]
        search_space['num_encoder_layers'] = [2, 3, 4]
        search_space['nhead'] = [4, 8]
        search_space['dim_feedforward'] = [512, 1024, 2048]
    
    # 랜덤 샘플링
    params = {}
    params['learning_rate'] = random.uniform(1e-5, 1e-3)
    params['batch_size'] = random.choice(search_space['batch_size'])
    params['sequence_length'] = random.choice(search_space['sequence_length'])
    params['dropout'] = random.choice(search_space['dropout'])
    params['normalization_method'] = random.choice(search_space['normalization_method'])

    if(model_name.lower() == 'cnntrans'):
        params['d_model'] = random.choice(search_space['d_model'])
        params['num_encoder_layers'] = random.choice(search_space['num_encoder_layers'])
        params['nhead'] = random.choice(search_space['nhead'])

        # d_model은 nhead로 나누어 떨어져야 함
        while params['d_model'] % params['nhead'] != 0:
            params['nhead'] = random.choice(search_space['nhead'])
        
        params['dim_feedforward'] = random.choice(search_space['dim_feedforward'])

    else:
        params['sae_noise_factor'] = random.uniform(0, 0.1)
        params['prediction_weight'] = random.uniform(0, 1)
        params['reconstruction_weight'] = random.uniform(0, 1)
        params['directional_weight'] = random.uniform(0, 1)   
        
    return params

if __name__ == "__main__":
    main()
