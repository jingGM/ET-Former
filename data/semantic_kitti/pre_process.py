import pickle
from os.path import join

import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import glob
import yaml

from data.data_utils import unpack, SemanticLabels, downsample_label


class KITTIProcessor:
    def __init__(self, config_path, data_root, render=False, process_type=0):
        self.data_root = data_root
        self.cfg = yaml.safe_load(open(config_path, 'r'))
        self.render = render
        self.process_type = process_type
        self.remap_lut = self._get_lut_map()
        self.sequences = self.cfg["split"]["train"] + self.cfg["split"]["valid"]

    def _get_lut_map(self):
        # make lookup table for mapping
        maxkey = max(self.cfg['learning_map'].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.cfg['learning_map'].keys())] = list(self.cfg['learning_map'].values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = SemanticLabels.invalid  # map 0 to 'invalid'
        remap_lut[0] = SemanticLabels.empty  # only 'empty' stays 'empty'.
        return remap_lut

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

    def _read_kitti(self, path, dtype, do_unpack):
        bin = np.fromfile(path, dtype=dtype)  # Flattened array
        if do_unpack:
            bin = unpack(bin)
        return bin

    def run(self, batch, index):
        sequences = sorted(self.sequences)
        if batch != 1:
            each_batch = int(len(sequences) / batch)
            if index == 0:
                sequences = sequences[:each_batch]
            elif index == batch - 1:
                sequences = sequences[(batch - 1) * each_batch:]
            else:
                sequences = sequences[each_batch * index:each_batch * (1 + index)]
        print(sequences)

        for sequence in sequences:
            sequence_path = os.path.join(self.data_root, self.cfg["root"], "dataset", "sequences", "{:02d}".format(sequence))
            label_paths = sorted(glob.glob(os.path.join(sequence_path, "voxels", "*.label")))
            invalid_paths = sorted(glob.glob(os.path.join(sequence_path, "voxels", "*.invalid")))
            out_dir = os.path.join(self.data_root, self.cfg["output_dir"], "labels", "{:02d}".format(sequence))
            os.makedirs(out_dir, exist_ok=True)

            for i in tqdm(range(len(label_paths))):
                frame_id, extension = os.path.splitext(os.path.basename(label_paths[i]))

                current_label = self._read_kitti(label_paths[i], dtype=np.uint16, do_unpack=False).astype(np.float32)
                INVALID = self._read_kitti(invalid_paths[i], dtype=np.uint8, do_unpack=True)
                current_label = self.remap_lut[current_label.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
                current_label[np.isclose(INVALID, 1)] = SemanticLabels.invalid  # Setting to unknown all voxels marked on invalid mask...
                current_label = current_label.reshape(self.cfg["voxel_size"])

                for scale in self.cfg["downscaling"]:
                    file_name = frame_id + "_" + str(scale) + ".npy"
                    output_file_path = os.path.join(out_dir, file_name)
                    if not os.path.exists(output_file_path):
                        LABEL_ds = downsample_label(current_label, self.cfg["voxel_size"], scale)
                        np.save(output_file_path, LABEL_ds)



if __name__ == "__main__":
    processor = KITTIProcessor("/data/semantic_kitti/semantic-kitti.yaml")
    processor.run()