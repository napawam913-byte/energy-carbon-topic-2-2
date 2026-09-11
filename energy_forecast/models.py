"""模块用途：将历史窗口展平并直接输出多步多目标预测的两层 MLP。"""
import torch
from .data import validate_config


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        c = validate_config(config)
        self.horizon, self.targets = c['horizon'],len(c['target_columns'])
        width = c['hidden_size']
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(c['lookback']*len(c['input_columns']),width),torch.nn.ReLU(),
            torch.nn.Linear(width,width),torch.nn.ReLU(),
            torch.nn.Linear(width,self.horizon*self.targets))

    def forward(self, x):
        return self.layers(x.flatten(start_dim=1)).reshape(-1,self.horizon,self.targets)
