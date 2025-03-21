import torch
import torch.nn as nn


# reference https://github.com/yuxumin/PoinTr, modified by Ricardo Cardoso


class FoldingNet(nn.Module):
    def __init__(self, num_pred, encoder_channel):
        super().__init__()
        self.num_pred = num_pred
        self.encoder_channel = encoder_channel
        self.grid_size = int(pow(self.num_pred,0.5) + 0.5)

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.encoder_channel + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(self.encoder_channel + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        a = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda() # 1 2 N
    
    def forward(self, feature_global):
        num_sample = self.grid_size * self.grid_size
        batch_size = feature_global.size(0)
        features = feature_global.view(batch_size, self.encoder_channel, 1).expand(batch_size, self.encoder_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(batch_size, 2, num_sample).to(feature_global.device)

        feature_global = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(feature_global)
        feature_global = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(feature_global)

        fd2 = fd2.transpose(2,1).contiguous()
        return fd2
