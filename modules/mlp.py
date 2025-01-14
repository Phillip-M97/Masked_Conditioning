# MLP for predicting the drag coefficient

import torch
import torch.nn as nn

class MLP(nn.Module):
    ''' MLP to predict drag coefficient of a vehicle given as reference point coordinates.
    '''
    def __init__(self) -> None:
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(42, 20),
            nn.BatchNorm1d(20),
            nn.SELU(),
            nn.Dropout(0.02),
            nn.Linear(20, 25),
            nn.BatchNorm1d(25),
            nn.SELU(),
            nn.Dropout(0.02),
            nn.Linear(25, 36),
            nn.BatchNorm1d(36),
            nn.SELU(),
            nn.Dropout(0.02),
            nn.Linear(36, 40),
            nn.BatchNorm1d(40),
            nn.SELU(),
            nn.Dropout(0.02),
            nn.Linear(40, 1)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)