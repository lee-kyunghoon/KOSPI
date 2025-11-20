"""
Warmup + Cosine Annealing 스케줄러 시각화
"""

import torch
import matplotlib.pyplot as plt
import yaml

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_scheduler():
    config = load_config()
    
    warmup_epochs = config['training']['warmup_epochs']
    total_epochs = config['training']['num_epochs']
    warmup_lr_start = config['training']['min_lr']
    max_lr = config['training']['learning_rate']
    min_lr = config['training']['min_lr']
    
    def get_lr(epoch):
        if epoch < warmup_epochs:
            # Warmup: linearly increase from min_lr to max_lr
            return warmup_lr_start + (max_lr - warmup_lr_start) * epoch / warmup_epochs
        else:
            # Cosine Annealing: decay from max_lr to min_lr
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))
    
    # Learning rate 계산
    epochs = list(range(total_epochs))
    learning_rates = [get_lr(epoch).item() if isinstance(get_lr(epoch), torch.Tensor) else get_lr(epoch) 
                     for epoch in epochs]
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, learning_rates, linewidth=2, color='blue')
    plt.axvline(x=warmup_epochs, color='red', linestyle='--', label=f'Warmup End (Epoch {warmup_epochs})')
    plt.axhline(y=max_lr, color='green', linestyle='--', alpha=0.5, label=f'Max LR: {max_lr:.2e}')
    plt.axhline(y=min_lr, color='orange', linestyle='--', alpha=0.5, label=f'Min LR: {min_lr:.2e}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Warmup + Cosine Annealing Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 저장
    plt.savefig('scheduler_visualization.png', dpi=150)
    print(f"✓ Scheduler visualization saved to 'scheduler_visualization.png'")
    
    # 주요 epoch의 learning rate 출력
    print("\n" + "="*60)
    print("Learning Rate Schedule")
    print("="*60)
    print(f"Epoch 0 (Start):       LR = {learning_rates[0]:.2e}")
    print(f"Epoch {warmup_epochs} (Warmup End): LR = {learning_rates[warmup_epochs]:.2e}")
    print(f"Epoch {total_epochs//2} (Middle):     LR = {learning_rates[total_epochs//2]:.2e}")
    print(f"Epoch {total_epochs-1} (End):        LR = {learning_rates[total_epochs-1]:.2e}")
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    visualize_scheduler()
