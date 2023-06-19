import random

import torch
import torch.nn as nn


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_feature(features,snumbe,seed):
    """

    Parameters
    ----------
    features: the features of point clouds, [B,C,N,D]
    snumbe: the number of sample features
    seed: random seed

    Returns
    -------
    sampled_features: the samples features of point clouds under different domain, [B,C,snumbe,D]

    """
    torch.manual_seed(seed)
    device = features.device
    B, C, N, _ = features.shape
    idx = torch.randint(0, N, (B, C, snumbe)).long()
    batch_view = [B, 1, 1]
    batch_repeat = [1, C, snumbe]
    row_view = [1, C, 1]
    row_repeat = [B, 1, snumbe]
    batch_indices = torch.arange(B, dtype=torch.long).view(batch_view).repeat(batch_repeat)
    row_indices = torch.arange(C, dtype=torch.long).view(row_view).repeat(row_repeat)
    sampled_features = features[batch_indices, row_indices, idx, :].view(B*C, snumbe,-1)
    return sampled_features


# def sample_and_group(xyz, points=None, npoint=256, radius=0.2, nsample=4, seed=6):
#     """
#     Input:
#         npoint:
#         radius:
#         nsample:
#         xyz: input points position data, [B, N, 3]
#         points: input points data, [B, N, D]
#     Return:
#         new_xyz: sampled points position data, [B, npoint, nsample, 3]
#         new_points: sampled points data, [B, npoint, nsample, 3+D]
#     """
#     B,_, _ = points.shape
#     fps_idx = farthest_point_sample(xyz, npoint)# [B, npoint, C]
#     new_xyz = index_points(xyz, fps_idx)
#     core_points = index_points(points, fps_idx).reshape(B*npoint, -1)
#
#     idx = query_ball_point(radius, nsample, xyz, new_xyz)
#     grouped_points = index_points(points, idx)
#     # grouped_points_mean = torch.mean(grouped_points,2).reshape(B*npoint, -1)
#     #     return core_points, grouped_points_mean
#     sampled_features = sample_feature(grouped_points, nsample, seed)
#     return sampled_features

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()

    fps_idx = farthest_point_sample(xyz, npoint).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx




# def voxel_set_idx(pc, size=2):
#     B, N, _ = pc.shape
#     device = pc.device
#     min_set = torch.amin(pc, axis=1).to(device)
#     max_set = torch.amax(pc, axis=1).to(device)
#     D_set = ((max_set - min_set) / size + 0.001).to(device)
#     Min = min_set.unsqueeze(axis=1).repeat(1,N,1).to(device)
#     # .expand_dims(min_set, axis=1).repeat(N, axis=1)
#     # Hdis = np.expand_dims(D_set, axis=1).repeat(N, axis=1)
#     Hdis = D_set.unsqueeze(axis=1).repeat(1,N,1).to(device)
#
#     H_idx = torch.floor((pc - Min) / Hdis).to(device)
#
#     def get_idx(pc_idx):
#         idx = torch.arange(size * size * size).reshape([size, size, size]).to(device)
#         x = int(pc_idx[0])
#         y = int(pc_idx[1])
#         z = int(pc_idx[2])
#         id = idx[x, y, z]
#         return id
#
#     H_idx_new = []
#     for i in range(B):
#          H_idx_new.append(list(map(get_idx, H_idx[i,:,:])))
#     H_idx_new = torch.as_tensor(H_idx_new).to(device)
#     return H_idx_new
#
# def sample_feature_1d(features,snumbe,seed):
#     """
#
#     Parameters
#     ----------
#     features: the features of point clouds, [B,C,N,D]
#     snumbe: the number of sample features
#     seed: random seed
#
#     Returns
#     -------
#     sampled_features: the samples features of point clouds under different domain, [B,C,snumbe,D]
#
#     """
#     torch.manual_seed(seed)
#     device = features.device
#     B, N = features.shape
#     idx = torch.randint(0, B, (snumbe,)).long().to(device)
#
#     sampled_features = features[idx, :].unsqueeze(dim=0)
#     return sampled_features
#
# def id_with_feat(feat, idx, sample_n =4, seed = 6, size=2):
#     F_R = None
#     for b in range(feat.shape[0]):
#         feat_rand = None
#         for i in range(size * size * size):
#             feat_now = feat[b, :, :]
#             feat_1 = feat_now[torch.where(idx[b, :] == i)]
#             if feat_1.shape[0] > 4:
#                 if feat_rand is None:
#                     feat_rand = sample_feature_1d(feat_1, 4, 6)
#                 else:
#                     feat_rand = torch.cat([feat_rand, sample_feature_1d(feat_1, sample_n, seed)], dim=0)
#         if F_R is None:
#             F_R = feat_rand
#         else:
#             F_R = torch.cat((F_R, feat_rand), dim=0)
#
#     return F_R
#
#
#
# class Voxelization(nn.Module):
#     def __init__(self, resolution, normalize=True):
#         super().__init__()
#         self.r = int(resolution)
#         self.normalize = normalize
#
#     def forward(self, features, coords):
#         coords = coords.detach()
#         norm_coords = coords - coords.mean(2, keepdim=True)
#
#         norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
#         vox_coords = torch.round(norm_coords).to(torch.int32)
#         return F.avg_voxelize(features, vox_coords, self.r), coords
