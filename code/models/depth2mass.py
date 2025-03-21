import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityBlock(nn.Module):
    def __init__(self, emb_dims):
        super(DensityBlock, self).__init__()
        self.linear1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 1)
        self.dp = nn.Dropout()
        
        # Exponential function parameters
        self.a = 0.05809935762708783
        self.b = 4.309885529518155
        self.c = 0.05747969878947174

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp(x)
        x = self.linear3(x)
        x = self.density_inductive_bias(x)
        return x
    
    def density_inductive_bias(self, x):
        exp = self.a * torch.exp(-self.b * (x - self.c))
        density_correction = 1 / exp
        
        return density_correction


class VolumeBlock(nn.Module):
    def __init__(self, emb_dims):
        super(VolumeBlock, self).__init__()
        self.linear1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 1)
        self.dp = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp(x)
        x = F.relu(self.linear3(x)) + 1e-6
        return x


class Depth2Mass(nn.Module):
    def __init__(self, emb_dims, pc_in_dims, pc_model, pc_completion, pc_out_dims):
        super(Depth2Mass, self).__init__()

        self.pc_completion = pc_completion

        if pc_model == 'none':
            self.model_pc = None
        if pc_model == 'pointnet':
            from models.pointnet import PointNet
            self.model_pc = PointNet(emb_dims)
        if pc_model == 'dgcnn':
            from models.dgcnn import DGCNN
            self.model_pc = DGCNN(emb_dims)
        if pc_model == 'point_transformer':
            from models.point_transformer import PointTransformer
            self.model_pc = PointTransformer(emb_dims, pc_in_dims, 3)
        
        from models.densenet import DenseNet
        if self.model_pc == None:
            self.model_densenet = DenseNet(2 * emb_dims)
        else:
            self.model_densenet = DenseNet(emb_dims)
        
        if self.pc_completion:
            from models.foldingnet import FoldingNet
            self.model_foldingnet = FoldingNet(pc_out_dims, emb_dims)
        
        self.densityBlock = DensityBlock(2 * emb_dims)
        self.volumeBlock = VolumeBlock(2 * emb_dims)

        self.c = 0.0601060


    def forward(self, x):
        pc, img = x
        img = img.squeeze(1)
        emb_img = self.model_densenet(img) # (B, emb_dims)
        if self.model_pc != None:
            emb_pc = self.model_pc(pc) # (B, emb_dims)
            emb_total = torch.cat((emb_img, emb_pc), dim=1) # (B, 2 * emb_dims)
        else:
            emb_total = emb_img

        density = self.densityBlock(emb_total) # (B, 1)
        density = self.c * density
        volume = self.volumeBlock(emb_total) # (B, 1)
        volume = (1.0 / self.c) * volume

        mass = density * volume

        if self.pc_completion:
            pc_reconstructed = self.model_foldingnet(emb_pc) # emb_pc[:, :512]
            return mass, pc_reconstructed
        
        return mass
