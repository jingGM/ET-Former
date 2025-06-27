from models.commons import BaseModel
from models.triplane.triplane_down_sample import TriplaneDownSample
from models.voxformer.resnet import ResNet
from models.voxformer.image_necks import FPN
from utilities.kitti_configs import DataDict
from utilities.functions import to_device


class QueryNet(BaseModel):
    def __init__(self, config, scale, data_type, model_type, device="cpu"):
        super().__init__(config, data_type, model_type, device=device)

        self.img_backbone = ResNet(depth=config.img_backbone.depth, num_stages=config.img_backbone.num_stages,
                                   out_indices=config.img_backbone.out_indices,
                                   frozen_stages=config.img_backbone.frozen_stages,
                                   norm_cfg=config.img_backbone.norm_cfg, norm_eval=config.img_backbone.norm_eval,
                                   style=config.img_backbone.style, stage_with_dcn=config.img_backbone.stage_with_dcn)
        self.img_neck = FPN(in_channels=config.img_neck.in_channels, out_channels=config.img_neck.out_channels,
                            start_level=config.img_neck.start_level, add_extra_convs=config.img_neck.add_extra_convs,
                            num_outs=config.img_neck.num_outs,
                            relu_before_extra_convs=config.img_neck.relu_before_extra_convs)

        self.triplane = TriplaneDownSample(config.triplane, scale, data_type, model_type, device)

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        if img.dim() == 5 and img.size(0) == 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        # if self.with_img_neck:
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward(self, data_dict):
        img_feats = self.extract_img_feat(img=to_device(data_dict[DataDict.img], device=self.device))

        voxel = self.triplane(img_feats=img_feats, proposal=None, queries=data_dict[DataDict.queries].to(int),
                              projection_matrix=data_dict[DataDict.w2i], pc_range=data_dict[DataDict.pc_range])
        return voxel
