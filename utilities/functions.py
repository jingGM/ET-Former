import collections
import math
import random
import warnings
from itertools import repeat

import numpy as np
from typing import Union
import torch
from numba import njit, prange
from torch import nn
from torch.nn import init

from utilities.configs import DatasetTypes


def hierarchical_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.ModuleList) or isinstance(m, nn.Sequential):
        for submodule in m:
            hierarchical_init(submodule)
    elif isinstance(m, nn.Module):
        for submodule in m.children():
            hierarchical_init(submodule)


def get_device(device: Union[torch.device, str] = "cuda") -> torch.device:
    """
    get the device of the input string or torch.device
    Args:
        device: string or torch device

    Returns:
        the device format that can be used for torch tensors
    """
    if isinstance(device, str):
        assert device == "cuda" or device == "cuda:0" or device == "cuda:1" or device == "cpu", \
            "device should only be 'cuda' or 'cpu' "
    device = torch.device(device)
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def to_device(x, device):
    """
    put the input to a torch tensor in the given device
    Args:
        x: list, tuple of torch tensors, dict of torch tensors or torch tensors
        device: pytorch device
    Returns:
        the input data in the given device
    """
    if isinstance(x, list):
        x = [to_device(item, device) for item in x]
    elif isinstance(x, tuple):
        x = (to_device(item, device) for item in x)
    elif isinstance(x, dict):
        x = {key: to_device(value, device) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if device == "cuda":
            x = x.cuda()
        else:
            x = x.to(device)
    return x


def release_cuda(x):
    """
    put the torch tensors from cuda to numpy
    Args:
        x: in put torch tensor, list of, dict of or tuple of torch tensors in cuda

    Returns:
        input data in local numpy
    """
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x


def inverse_transform(pts, transformation):
    if len(pts.shape) == 1:
        pts = pts[None, :]
    elif len(pts.shape) == 2:
        pass
    else:
        raise Exception("points shape is not correct")
    if pts.shape[-1] == 2:
        pts = np.concatenate((pts, np.ones_like(pts[:, :1]) * transformation[2, -1]), axis=-1)
    if transformation.shape[0] == 3:
        last_vector = np.zeros(4)[None, :]
        last_vector[0, -1] = 1
        transformation = np.concatenate((transformation, last_vector), axis=0)
    inv_transformation = np.linalg.inv(transformation)
    new_pts = appy_tranformation(pts, inv_transformation)
    return new_pts


def appy_tranformation(points, transform):
    assert transform.shape[1] == 4 and transform.shape[0] >= 3, "transform shape should be (3,4) or (4,4)"
    assert points.shape[1] == 3, "points shape should be (n,3)"
    if len(points.shape) == 2:
        rotation = transform[:3, :3]  # (3, 3)
        translation = transform[None, :3, 3]  # (1, 3)
        points = np.matmul(points, rotation.transpose(-1, -2)) + translation
    else:
        raise Exception("points shape is not correct: Nx3 or Nx3x4")
    return points


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])
    # sensor_pose = 10

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    coords_grid[:, 0] = temp[:, 1]
    coords_grid[:, 1] = temp[:, 0]
    return coords_grid


@njit(parallel=True)
def vox2world(vol_origin, vox_coords, vox_size, offsets=(0.5, 0.5, 0.5)):
    """Convert voxel grid coordinates to world coordinates."""
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)

    for i in prange(vox_coords.shape[0]):
        for j in range(3):
            cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j]) + vox_size * offsets[j]
    return cam_pts


@njit(parallel=True)
def pts2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates."""
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
        pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
        pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix


def vox2pix(cam_E, cam_k, vox_origin, voxel_size, img_W, img_H, scene_size):
    """
    compute the 2D projection of voxels centroids

    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2

    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = vox_origin
    vol_bnds[:, 1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
    vox_coords = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = appy_tranformation(points=cam_pts, transform=cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = pts2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0, np.logical_and(pix_x < img_W, np.logical_and(pix_y >= 0, np.logical_and(pix_y < img_H, pix_z > 0))))

    return projected_pix, fov_mask, pix_z


def compute_CP_mega_matrix(target, is_binary=False):
    """
    Parameters
    ---------
    target: (H, W, D)
        contains voxels semantic labels

    is_binary: bool
        if True, return binary voxels relations else return 4-way relations
    """
    label = target.reshape(-1)
    label_row = label
    N = label.shape[0]
    super_voxel_size = [i // 2 for i in target.shape]
    if is_binary:
        matrix = np.zeros((2, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)
    else:
        matrix = np.zeros((4, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)

    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                label_col_megas = np.array([
                    target[xx * 2, yy * 2, zz * 2],
                    target[xx * 2 + 1, yy * 2, zz * 2],
                    target[xx * 2, yy * 2 + 1, zz * 2],
                    target[xx * 2, yy * 2, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2, zz * 2 + 1],
                    target[xx * 2, yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ])
                label_col_megas = label_col_megas[label_col_megas != 255]
                for label_col_mega in label_col_megas:
                    label_col = np.ones(N) * label_col_mega
                    if not is_binary:
                        matrix[0, (label_row != 255) & (label_col == label_row) & (
                                    label_col != 0), col_idx] = 1.0  # non non same
                        matrix[1, (label_row != 255) & (label_col != label_row) & (label_col != 0) & (
                                    label_row != 0), col_idx] = 1.0  # non non diff
                        matrix[2, (label_row != 255) & (label_row == label_col) & (
                                    label_col == 0), col_idx] = 1.0  # empty empty
                        matrix[3, (label_row != 255) & (label_row != label_col) & (
                                    (label_row == 0) | (label_col == 0)), col_idx] = 1.0  # nonempty empty
                    else:
                        matrix[0, (label_row != 255) & (label_col != label_row), col_idx] = 1.0  # diff
                        matrix[1, (label_row != 255) & (label_col == label_row), col_idx] = 1.0  # same
    return matrix


def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
    valid_pix = np.logical_and(pix_x >= min_x,
                               np.logical_and(pix_x < max_x,
                                              np.logical_and(pix_y >= min_y,
                                                             np.logical_and(pix_y < max_y,
                                                                            pix_z > 0))))
    return valid_pix


def compute_local_frustums(projected_pix, pix_z, target, img_W, img_H, dataset, n_classes, size=4):
    """
    Compute the local frustums mask and their class frequencies

    Parameters:
    ----------
    projected_pix: (N, 2)
        2D projected pix of all voxels
    pix_z: (N,)
        Distance of the camera sensor to voxels
    target: (H, W, D)
        Voxelized sematic labels
    img_W: int
        Image width
    img_H: int
        Image height
    dataset: str
        ="NYU" or "kitti" (for both SemKITTI and KITTI-360)
    n_classes: int
        Number of classes (12 for NYU and 20 for SemKITTI)
    size: int
        determine the number of local frustums i.e. size * size

    Returns
    -------
    frustums_masks: (n_frustums, N)
        List of frustums_masks, each indicates the belonging voxels
    frustums_class_dists: (n_frustums, n_classes)
        Contains the class frequencies in each frustum
    """
    H, W, D = target.shape
    ranges = [(i * 1.0 / size, (i * 1.0 + 1) / size) for i in range(size)]
    local_frustum_masks = []
    local_frustum_class_dists = []
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    for y in ranges:
        for x in ranges:
            start_x = x[0] * img_W
            end_x = x[1] * img_W
            start_y = y[0] * img_H
            end_y = y[1] * img_H
            local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y, pix_z)
            if dataset == DatasetTypes.nyu:
                mask = (target != 255) & local_frustum.reshape(60, 60, 36) #np.moveaxis(, [0, 1, 2], [0, 2, 1])
            elif dataset == DatasetTypes.kitti:
                mask = (target != 255) & local_frustum.reshape(H, W, D)

            local_frustum_masks.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes)
            class_counts[classes.astype(int)] = cnts
            local_frustum_class_dists.append(class_counts)
    frustums_masks, frustums_class_dists = np.array(local_frustum_masks), np.array(local_frustum_class_dists)
    return frustums_masks, frustums_class_dists


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
