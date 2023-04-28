import torch.optim as optim

# Learning Rate Scheduler introduced in the Original Paper
# Adjusts the lr during the training to help the moder converge
# and achieve better performance
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model

        # Warmup steps prevents the learning rate from being too high in the beggining
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def learning_rate(self):
        return self.d_model ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'current_step': self.current_step,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
        self.current_step = state_dict['current_step']
    

def create_optimizer_and_scheduler(model, d_model, warmup_steps, init_lr, weight_decay=0.0, original=False):
    # Not original Adam Optimizer
    # Using Adam with Weight Decay and AMSGRAD
    # AMSGRAD is a variant of the Adam Optimizer that adresses the issue of convergence in certain settings
    # Maintains a maximum of past squared gradients, ensuring better convergence properties

    # Weight Decay prevents overfitting
    # Adds a penalty term to the loss function
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay, amsgrad=True)

    # In case we want to follow the original paper
    if original : optimizer = optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model, warmup_steps)
    return optimizer, scheduler