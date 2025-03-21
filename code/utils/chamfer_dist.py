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


def chamfer_distance(xyz1, xyz2):
    # input: xyz1: (B, N1, 3) xyz2: (B, N2, 3) -> output: dist1: (B, N1) dist2: (B, N2)

    # Perform KNN with the xyz1 point cloud to the xyz2 point cloud
    dists_12 = square_distance(xyz1, xyz2) # (B, N1, N2)
    idx_12 = dists_12.argsort()[:, :, :1]  # (B, N1, 1)
    torch.cuda.empty_cache()
    grouped_xyz_12 = index_points(xyz2, idx_12) # (B, N1, 1, 3)
    torch.cuda.empty_cache()
    grouped_xyz_12 = grouped_xyz_12.squeeze(dim = 2) # (B, N1, 3)

    # Compute the L2 distance of the nearest neighbor in xyz2 to the corresponding xyz1 point
    grouped_xyz_12_norm = grouped_xyz_12 - xyz1 # (B, N1, 3)
    torch.cuda.empty_cache()
    norm_12 = torch.norm(grouped_xyz_12_norm, dim=2) # (B, N1)
    torch.cuda.empty_cache()

    # Compute the average along the second dimension
    mean_12 = torch.mean(norm_12, dim=1, keepdim=True)
    torch.cuda.empty_cache()

    return mean_12

def chamfer_distance_bidirectional(xyz1, xyz2):
    # input: xyz1: (B, N1, 3) xyz2: (B, N2, 3) -> output: dist1: (B, N1) dist2: (B, N2)

    chanf_dist_12 = chamfer_distance(xyz1, xyz2)
    chanf_dist_21 = chamfer_distance(xyz2, xyz1)

    # Compute the average of the sum
    avg_dist = torch.mean(chanf_dist_12 + chanf_dist_21)

    return avg_dist
