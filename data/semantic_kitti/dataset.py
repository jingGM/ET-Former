import glob
import os
import pickle
from os.path import join

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from numpy.linalg import inv
from scipy import stats
from typing import Dict

import yaml
from torch.utils.data import Dataset
from torchvision import transforms

from utilities.configs import DatasetUsages, DataDict,  QueryTypes, ModelTypes
from utilities.functions import vox2pix


class KittiDataset(Dataset):
    def __init__(self, config_path, root, dataset_usage, model_type, scale=1, render=False):
        self.dataset_usage = dataset_usage
        self.cfg: Dict = yaml.safe_load(open(config_path, 'r'))
        self.root = root
        self.render = render
        self.model_type = model_type

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.target_scale = scale
        if self.model_type == ModelTypes.query:
            self.target_scale = 2

        self.data_root = self.root
        self.sequence_root = join(self.data_root, "dataset", "sequences")
        self.label_root = join(self.root, self.cfg["output_dir"], "labels")
        self.voxel_root = join(self.root, self.cfg["output_dir"], "voxel")

        self.n_classes = len(self.cfg["labels"].keys())
        self.sequences = self.cfg["split"][dataset_usage]

        self.frustum_size = self.cfg["frustum_size"]
        self.project_scale = self.cfg["project_scale"]
        self.scene_size = self.cfg["scene_size"]
        self.vox_origin = np.array(self.cfg["vox_origin"], dtype=float)
        self.output_scale = int(self.project_scale / 2)

        self.voxel_size = self.cfg["voxel_grid"]  # 0.2m
        self.image_size = self.cfg["image_size"]  # 0.2m
        self.voxel_dimension = np.array(self.cfg["voxel_size"], dtype=int)  # 0.2m

        self.query_tag = "query_iou5203_pre7712_rec6153"
        self.depthmodel = "msnet3d"
        self.nsweep = str(10)

        self.scans = []
        self._get_scan_data()

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    def _get_scan_data(self):
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(join(self.sequence_root, "{:02d}".format(sequence), "calib.txt"))
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]

            proj_matrix = P @ T_velo_2_cam

            query_global_path = os.path.join(
                self.data_root, "dataset", "sequences_" + self.depthmodel + "_sweep" + self.nsweep,
                "{:02d}".format(sequence), "queries", "*." + self.query_tag
            )
            depth_global_path = os.path.join(
                self.data_root, "dataset", "sequences_" + self.depthmodel + "_sweep" + self.nsweep,
                "{:02d}".format(sequence), "voxels"
            )
            query_paths = glob.glob(query_global_path)

            for f_id in range(len(query_paths)):

                filename = os.path.basename(query_paths[f_id])
                # print("frame ID: ", filename)
                frame_id = os.path.splitext(filename)[0]

                self.scans.append(
                    {
                        "sequence": "{:02d}".format(sequence),
                        DataDict.P: P,
                        DataDict.w2c: T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": join(self.sequence_root, "{:02d}".format(sequence), "voxels", frame_id+".bin"),
                        "proposal_path": query_paths[f_id],
                        "depth_path": join(depth_global_path, frame_id+".pseudo"),
                    }
                )

    def __len__(self):
        return len(self.scans)

    def _load_data(self, T_velo_2_cam, cam_k, proposal_path):
        with open(proposal_path, "rb") as handle:
            proposal_bin = pickle.load(handle)
        projected_pix, fov_mask, pix_z = vox2pix(
            T_velo_2_cam,
            cam_k,
            self.vox_origin,
            self.voxel_size,
            self.image_size[0],
            self.image_size[1],
            self.scene_size,
        )

        data = {
            DataDict.proposal: proposal_bin,
            "fov_mask_1": fov_mask
        }
        return data

    def _read_query(self, path, dtype):
        bin = np.fromfile(path, dtype=dtype)  # Flattened array

        uncompressed = np.zeros(bin.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = bin[:] >> 7 & 1
        uncompressed[1::8] = bin[:] >> 6 & 1
        uncompressed[2::8] = bin[:] >> 5 & 1
        uncompressed[3::8] = bin[:] >> 4 & 1
        uncompressed[4::8] = bin[:] >> 3 & 1
        uncompressed[5::8] = bin[:] >> 2 & 1
        uncompressed[6::8] = bin[:] >> 1 & 1
        uncompressed[7::8] = bin[:] & 1
        return uncompressed

    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed

    def read_SemKITTI(self, path, dtype, do_unpack):
        bin = np.fromfile(path, dtype=dtype)  # Flattened array
        if do_unpack:
            bin = self.unpack(bin)
        return bin

    def __getitem__(self, index):
        scan = self.scans[index]

        sequence = scan["sequence"]
        filename = os.path.basename(scan["voxel_path"])
        frame_id = os.path.splitext(filename)[0]

        data = {}

        P = scan[DataDict.P]
        T_velo_2_cam = scan[DataDict.w2c]
        cam_k = P[0:3, 0:3]
        data[DataDict.w2c] = T_velo_2_cam
        data[DataDict.P] = P
        data[DataDict.intrinsic] = cam_k

        viewpad = np.eye(4)
        viewpad[:cam_k.shape[0], :cam_k.shape[1]] = cam_k
        # transform 3d point in lidar coordinate to 2D image (projection matrix)
        lidar2img_rt = (viewpad @ T_velo_2_cam)
        data[DataDict.w2i] = torch.from_numpy(lidar2img_rt).unsqueeze(0)
        data[DataDict.vox_origin] = self.vox_origin
        end_range = np.array(self.vox_origin) + np.array(self.scene_size)
        data[DataDict.pc_range] = np.array(self.vox_origin.tolist() + end_range.tolist())

        target = np.load(join(self.label_root, sequence, frame_id + "_{}.npy".format(self.target_scale)))

        rgb_path = join(self.sequence_root, sequence, "image_2", frame_id + ".png")
        image = Image.open(rgb_path).convert("RGB")

        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.0
        if self.dataset_usage != DatasetUsages.train:
            data[DataDict.pil_img] = image
        image = image[:self.image_size[1], :self.image_size[0], :]  # crop image

        if self.model_type == ModelTypes.query:
            pseudo_pc_bin = self.read_SemKITTI(scan["voxel_path"], dtype=np.uint8, do_unpack=True).astype(np.float32)
            data[DataDict.queries] = np.reshape(pseudo_pc_bin, tuple(self.voxel_dimension.tolist()))

            proposal_path = scan["proposal_path"]
            proposal_bin = self._read_query(proposal_path, dtype=np.uint8).astype(np.float32)
            vox_dim = self.voxel_dimension / 2
            data[DataDict.proposal] = np.reshape(proposal_bin, vox_dim.astype(int))

            data[DataDict.voxel_dimension] = (self.voxel_dimension / 2).astype(int)
            target = torch.from_numpy(target)
            ones = torch.ones_like(target)
            target = torch.where(torch.logical_or(target == 255, target == 0), target, ones)  # [1, 128, 128, 16]
            data[DataDict.img] = self.normalize_rgb(image)
            data[DataDict.name] = join(sequence, frame_id)

        elif self.model_type == ModelTypes.cvae or self.model_type == ModelTypes.triplane:
            query_path = join(self.data_root, "dataset/sequences_queries", sequence, frame_id + ".pkl")
            # if self.query_type == QueryTypes.mono_input:
            #     query_path = join(self.data_root, "dataset/sequences_queries", sequence, frame_id + ".pkl")
            # else:
            #     query_path = scan["proposal_path"]
            gsdo_data = self._load_data(T_velo_2_cam, cam_k, proposal_path=query_path)
            data.update(gsdo_data)
            data[DataDict.img] = self.normalize_rgb(image).unsqueeze(0)
            data[DataDict.name] = (sequence, frame_id)
        else:
            raise Exception("the model type is not defined")

        data[DataDict.target] = target
        return data


# if __name__ == "__main__":
#     from utilities.configs import DatasetConfig, ModelTypes
#
#     dataset = KittiDataset(
#         config_path="data/semantic_kitti/semantic-kitti.yaml",
#         root="/media/jing/update/ubuntu/work/2024/",
#         # root="/home/ANT.AMAZON.COM/jliangmd/Documents/dataset",
#         dataset_usage=DatasetUsages.train, render=False, model_type=ModelTypes.query, query_type=QueryTypes.mono_input)
#     data_ = dataset[0]
#     print("test")
