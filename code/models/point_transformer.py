import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# reference https://github.com/yanx27/Pointnet_Pointnet2_pytorch and
# https://github.com/qq456cvb/Point-Transformers, modified by Ricardo Cardoso


def index_points(points, idx):
    # input: points: (B, N, C) idx: (B, S, [K]) -> output: (B, S, [K], C)
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def square_distance(src, dst):
    # input: src (B, N, C) dst (B, M, C) -> output: dist (B, N, M)
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def farthest_point_sample(xyz, n_points):
    # input: xyz: (B, N, 3) -> output: centroids: (B, n_points)
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, n_points, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]

    return centroids


def sample_and_group(n_points_fps, k, xyz, features):
    # input: xyz: (B, N, 3) points: (B, N, D) -> output: new_xyz: (B, npoint, nsample, 3) new_points: (B, npoint, nsample, 3 + D)
    B, N, C = xyz.shape
    S = n_points_fps

    # Perform furthest point sampling
    fps_idx = farthest_point_sample(xyz, n_points_fps) # (B, n_points_fps)
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()

    # Perform KNN with the fps points to the original point cloud
    dists = square_distance(new_xyz, xyz)  # (B, n_points_fps, N)
    idx = dists.argsort()[:, :, :k]  # (B, n_points_fps, k)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # (B, n_points_fps, k, C)
    torch.cuda.empty_cache()

    # Compute the relative position of the k nearest neighbors to the corresponding fps point
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()


    if features is not None:
        # Get features for the k nearest neighbors of each fps point
        grouped_points = index_points(features, idx)

        # Concatenate the relative position and the features
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # (B, n_points_fps, k, C+D)
    else:
        new_points = grouped_xyz_norm

    # return the fps points and the relative position of the knn points for each fps point (with or without features)
    return new_xyz, new_points


class TransformerBlock(nn.Module):
    def __init__(self, features_dim, internal_dim, k):
        super().__init__()

        self.fc1 = nn.Linear(features_dim, internal_dim)
        self.fc2 = nn.Linear(internal_dim, features_dim)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, internal_dim)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, internal_dim)
        )

        self.w_query = nn.Linear(internal_dim, internal_dim, bias=False)
        self.w_keys = nn.Linear(internal_dim, internal_dim, bias=False)
        self.w_values = nn.Linear(internal_dim, internal_dim, bias=False)

        self.k = k
        
    # xyz: (b, n, 3), features: (b, n, features_dim)
    def forward(self, xyz, features):
        # calculate the distance matrix
        dists = square_distance(xyz, xyz)

        # get the first k nearest neighbors
        knn_idx = dists.argsort()[:, :, :self.k]  # (b, n, k)
        knn_xyz = index_points(xyz, knn_idx) # (b, n, k, 3)
        
        features_pre = features

        # Convert the features to the internal dimension
        x = self.fc1(features) # (b, n, internal_dim)

        # Compute the query for each point (aka phi)
        query = self.w_query(x) # b x n x internal_dim

        # Compute the keys and values for each k nearest neighbor of each point
        keys = index_points(self.w_keys(x), knn_idx) # b x n x k x internal_dim (aka psi)
        values = index_points(self.w_values(x), knn_idx) # b x n x k x internal_dim (aka alpha)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x internal_dim
        
        attn = self.fc_gamma(query[:, :, None] - keys + pos_enc)  # b x n x k x internal_dim
        # Softamx applied on the k dimension
        attn = F.softmax(attn / np.sqrt(keys.size(-1)), dim=-2)  # b x n x k x internal_dim
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, values + pos_enc)
        res = self.fc2(res) + features_pre
        return res, attn


class TransitionDown(nn.Module):
    def __init__(self, n_points_fps, k, channel_dims):
        super(TransitionDown, self).__init__()

        self.n_points_fps = n_points_fps
        self.k = k

        self.conv1 = nn.Conv2d(channel_dims[0], channel_dims[1], 1)
        self.conv2 = nn.Conv2d(channel_dims[1], channel_dims[2], 1)
        self.bn1 = nn.BatchNorm2d(channel_dims[1])
        self.bn2 = nn.BatchNorm2d(channel_dims[2])
    
    def forward(self, xyz, features):
        # input: xyz: (B, N, C) points: (B, N, f) -> output: new_xyz: (B, n_points_fps, k, C) new_points: (B, n_points_fps, k, C + f)

        # Get furthest point sampling and relative position of knn for each fps point with concatenation of features
        # new_xyz: (B, n_points_fps, C)
        # new_points: (B, n_points_fps, k, C+f)
        fps_xyz, fps_knn_xyz = sample_and_group(self.n_points_fps, self.k, xyz, features)

        x = fps_knn_xyz.permute(0, 3, 2, 1) # (B, C+f, k, n_points_fps)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2)[0].transpose(1, 2)

        return fps_xyz, x
        

class PointTransformer(nn.Module):
    def __init__(self, emb_dims, n_points_pc, dim_pc, internal_dim=512, k=20):
        super().__init__()

        self.emb_dims = emb_dims

        self.fc1 = nn.Sequential(
            nn.Linear(dim_pc, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # nneighbor: 16, nblocks: 4, transformer_dim: 512

        # features_dim, internal_dim, k
        self.tfblock1 = TransformerBlock(32, internal_dim, k)
        self.tfblock2 = TransformerBlock(64, internal_dim, k)
        self.tfblock3 = TransformerBlock(128, internal_dim, k)
        self.tfblock4 = TransformerBlock(256, internal_dim, k)
        self.tfblock5 = TransformerBlock(512, internal_dim, k)
        self.trdown1 = TransitionDown(n_points_pc // 4, k, [32 + 3, 64, 64])
        self.trdown2 = TransitionDown(n_points_pc // 16, k, [64 + 3, 128, 128])
        self.trdown3 = TransitionDown(n_points_pc // 64, k, [128 + 3, 256, 256])
        self.trdown4 = TransitionDown(n_points_pc // 256, k, [256 + 3, 512, 512])
        if emb_dims == 1024:
            self.tfblock6 = TransformerBlock(1024, internal_dim, k)
            self.trdown5 = TransitionDown(n_points_pc // 512, k, [512 + 3, 1024, 1024])
    
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])

        xyz = x[..., :3]

        x = self.fc1(x)
        x = self.tfblock1(xyz, x)[0]
        xyz, x = self.trdown1(xyz, x)
        x = self.tfblock2(xyz, x)[0]
        xyz, x = self.trdown2(xyz, x)
        x = self.tfblock3(xyz, x)[0]
        xyz, x = self.trdown3(xyz, x)
        x = self.tfblock4(xyz, x)[0]
        xyz, x = self.trdown4(xyz, x)
        x = self.tfblock5(xyz, x)[0]
        if self.emb_dims == 1024:
            xyz, x = self.trdown5(xyz, x)
            x = self.tfblock6(xyz, x)[0]
        
        x = x.transpose(1, 2)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=2)
        return x
