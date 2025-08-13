import torch
from torch import nn
from utils.train_util import get_expon_lr_func

class Model(nn.Module):
    def __init__(self,
                 vertices: torch.Tensor,
                 indices: torch.Tensor,
                 max_sh_deg: int = 2):
        super().__init__()
        self.sh_dim = ((1+max_sh_deg)**2-1)*3
        self.register_buffer("vertices", vertices)          # immutable
        self.register_buffer("indices", indices.int())

        self.colors = nn.Parameter(0.5*torch.ones((len(self), 3)))
        self.alpha = nn.Parameter(0.5*torch.ones((len(self), 1)))
        self.sh_features = nn.Parameter(torch.zeros((len(self), self.sh_dim)))

    def __len__(self):
        return self.vertices.shape[0]

class TetOptimizer:
    def __init__(self,
                 model: Model,
                 alpha_lr: float = 1e-3,
                 final_alpha_lr: float = 1e-4,
                 color_lr: float = 1e-3,
                 final_color_lr: float = 1e-4,
                 sh_lr: float = 1e-3,
                 final_sh_lr: float = 1e-4,
                 freeze_start: int = 15000,
                 iterations: int = 30000,
                 **kwargs
    ) -> None:
        self.model = model
        self.alpha_optim = torch.optim.Adam([
            {"params": [model.colors],  "lr": color_lr,  "name": "colors"},
        ])
        self.color_optim = torch.optim.Adam([
            {"params": [model.alpha],      "lr": alpha_lr,    "name": "alpha"},
        ])
        self.sh_optim = torch.optim.Adam([
            {"params": [model.sh_features], "lr": sh_lr, "name": "sh"},
        ])
        self.freeze_start = freeze_start
        self.alpha_scheduler = get_expon_lr_func(
            lr_init=alpha_lr,
            lr_final=final_alpha_lr,
            lr_delay_mult=1,
            max_steps=iterations - self.freeze_start,
            lr_delay_steps=0)
        self.color_scheduler = get_expon_lr_func(
            lr_init=color_lr,
            lr_final=final_color_lr,
            lr_delay_mult=1,
            max_steps=iterations - self.freeze_start,
            lr_delay_steps=0)
        self.sh_scheduler = get_expon_lr_func(
            lr_init=sh_lr,
            lr_final=final_sh_lr,
            lr_delay_mult=1,
            max_steps=iterations - self.freeze_start,
            lr_delay_steps=0)
        self.vertex_optim = None  # geometry is frozen

    def update_triangulation(self, *_, **__):
        return None

    def step(self):
        self.alpha_optim.step()
        self.color_optim.step()
        self.sh_optim.step()

    def zero_grad(self):
        self.alpha_optim.zero_grad()
        self.color_optim.zero_grad()
        self.sh_optim.zero_grad()

    def main_step(self):
        self.step()

    def main_zero_grad(self):
        self.zero_grad()


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        for param_group in self.alpha_optim.param_groups:
            lr = self.alpha_scheduler(iteration - self.freeze_start)
            param_group['lr'] = lr
        for param_group in self.color_optim.param_groups:
            lr = self.color_scheduler(iteration - self.freeze_start)
            param_group['lr'] = lr
        for param_group in self.sh_optim.param_groups:
            lr = self.sh_scheduler(iteration - self.freeze_start)
            param_group['lr'] = lr
