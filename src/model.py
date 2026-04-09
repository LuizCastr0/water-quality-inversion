# model.py — salvar em fase2/src/model.py
import torch
import torch.nn as nn

class WaterQualityMLP(nn.Module):
    """
    Backbone compartilhado + duas cabeças independentes (turb, cha).
    Saída em escala log1p — a inversão (expm1) é feita na inferência.
    Dropout mais alto na cabeça cha porque tem 3x menos dados.
    """
    def __init__(self, in_features: int = 33, dropout_backbone: float = 0.3,
                dropout_cha: float = 0.5):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_backbone),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_backbone),
        )

        self.head_turb = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout_backbone),
            nn.Linear(32, 1),
        )

        self.head_cha = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout_cha),   # mais regularização — menos dados
            nn.Linear(32, 1),
        )

    def forward(self, x, target='both'):
        z = self.backbone(x)
        if target == 'turb':
            return self.head_turb(z).squeeze(-1)
        if target == 'cha':
            return self.head_cha(z).squeeze(-1)
        return self.head_turb(z).squeeze(-1), self.head_cha(z).squeeze(-1)