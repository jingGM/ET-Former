import numpy as np


KiTTi_Class_Names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)
KiTTi_Class_Weights = 1 / np.log(semantic_kitti_class_frequencies + 0.001)

SemanticColors = {0: [0, 0, 0],
                  1: [0, 0, 255],
                  2: [31, 119, 180],
                  3: [255, 127, 14],
                  4: [44, 160, 44],
                  5: [214, 39, 40],
                  6: [148, 103, 189],
                  7: [140, 86, 75],
                  8: [227, 119, 194],
                  9: [189, 189, 34],
                  10: [245, 150, 100],
                  11: [245, 230, 100],
                  12: [255, 187, 120],
                  13: [250, 80, 100],
                  14: [23, 190, 207],
                  15: [150, 60, 30],
                  16: [255, 0, 0],
                  18: [180, 30, 80],
                  20: [255, 255, 0],
                  30: [30, 30, 255],
                  31: [200, 40, 255],
                  32: [90, 30, 150],
                  40: [255, 0, 255],
                  44: [255, 150, 255],
                  48: [75, 0, 75],
                  49: [75, 0, 175],
                  50: [0, 200, 255],
                  51: [50, 120, 255],
                  52: [0, 150, 255],
                  60: [170, 255, 150],
                  70: [0, 175, 0],
                  71: [0, 60, 135],
                  72: [80, 240, 150],
                  80: [150, 240, 255],
                  81: [0, 0, 255],
                  99: [255, 255, 50],
                  255: [255, 255, 255],
                  }
colors = []
for i in range(256):
    if i in SemanticColors.keys():
        val = SemanticColors[i] + [255]
    else:
        val = [0,0,0,255]
    colors.append(val)
# print(colors)

class DataDict:
    name = "name"
    pose = "pose"
    P = "P"
    intrinsic = "intrinsic"
    w2c = "w2c"
    w2i = "w2i"
    c2w = "c2w"
    vox_origin = "vox_origin"
    voxel_dimension = "voxel_dimension"
    pc_range = "pc_range"
    proposal = "proposal"
    queries = "queries"

    target = "target"
    img = "img"
    pil_img = "pil_img"

    image_features = "image_features"
    voxel_features = "voxel_features"

    ssc_pred = "ssc_pred"
    P_logits = "P_logits"
    x = "x"
    mu = "mu"
    log_var = "log_var"

