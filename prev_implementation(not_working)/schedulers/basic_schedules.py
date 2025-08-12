import optax

def create_linear_schedule(init_lr: float, total_steps: int, end_lr: float = 0.0):
    """Create a linear decay schedule from init_lr to end_lr over total_steps."""
    return optax.linear_schedule(
        init_value=init_lr,
        end_value=end_lr,
        transition_steps=total_steps
    )

class KLAdaptiveLR:
    """KL divergence-based adaptive learning rate for PPO."""
    
    def __init__(self, init_lr: float = 3e-4, kl_target: float = 0.008, 
                 kl_factor: float = 2.0, lr_bounds: tuple = (1e-6, 1e-2)):
        self.init_lr = init_lr
        self.current_lr = init_lr
        self.kl_target = kl_target
        self.kl_factor = kl_factor
        self.lr_min, self.lr_max = lr_bounds
        
    def update(self, kl_divergence: float) -> float:
        """Update learning rate based on KL divergence."""
        if kl_divergence > self.kl_target * self.kl_factor:
            # KL too high, decrease LR
            self.current_lr = max(self.current_lr / 1.5, self.lr_min)
        elif kl_divergence < self.kl_target / self.kl_factor:
            # KL too low, increase LR  
            self.current_lr = min(self.current_lr * 1.1, self.lr_max)
        
        return self.current_lr
    
    def get_lr(self) -> float:
        return self.current_lr