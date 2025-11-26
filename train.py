import torch
import yaml
from trainers.aecnn_trainer import AECNNTrainer
from trainers.cnntrans_trainer import CNNTransTrainer


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config_path = 'config/config.yaml'
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = config['model']['model_name'].lower()
    
    if model_name == 'aecnn':
        trainer = AECNNTrainer(config, device, config_path)
    elif model_name == 'cnntrans':
        trainer = CNNTransTrainer(config, device, config_path)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'aecnn' or 'cnntrans'.")
    
    trainer.train()


if __name__ == "__main__":
    main()
