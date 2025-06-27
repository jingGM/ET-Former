import time
from warnings import warn

import torch
import torch.nn as nn
from einops import rearrange

from models.query_net import QueryNet
from models.triplane.triplane import Triplane
from models.voxformer.voxformer import VoxFormer
from utilities.configs import ModelTypes, DataDict
from utilities.functions import hierarchical_init


def get_model(config, data_type, device):
    return ETFORMER(config, data_type, device)


class ETFORMER(nn.Module):
    def __init__(self, cfgs, data_type, device="cpu"):
        super().__init__()
        self.cfgs = cfgs
        self.stage = self.cfgs.stage
        self.device = device
        self.data_type = data_type
        self.model_type = self.cfgs.type

        # stage 1
        if self.model_type == ModelTypes.query:
            self.baseline = QueryNet(config=self.cfgs.stage1, scale=self.cfgs.stage1.scale, data_type=self.data_type,
                                     model_type=self.model_type, device=self.device)
            self.output = nn.Sequential(nn.Linear(self.cfgs.stage1.triplane.pe_num_fts, self.cfgs.stage1.hidden_dims),
                                        nn.Softsign(),
                                        nn.Linear(self.cfgs.stage1.hidden_dims, self.cfgs.stage1.triplane.n_classes),
                                        nn.Softsign())
            hierarchical_init(self.output)

        # Stage 2:
        elif self.model_type == ModelTypes.triplane:
            self.baseline = VoxFormer(config=self.cfgs.vox_former, data_type=self.data_type, device=device,
                                      model_type=self.cfgs.type)
            self.triplane = Triplane(config=self.cfgs.triplane, data_type=self.data_type, device=device,
                                     model_type=self.cfgs.type)
            self.decoder = nn.Sequential(
                nn.Linear(self.cfgs.triplane.pe_num_fts, self.cfgs.triplane.hidden_dims), nn.Softplus(),
                nn.Linear(self.cfgs.triplane.hidden_dims, self.cfgs.triplane.n_classes * 8)
            )
            hierarchical_init(self.decoder)

        elif self.model_type == ModelTypes.cvae:
            self.baseline = VoxFormer(config=self.cfgs.vox_former, data_type=self.data_type, device=device,
                                      model_type=self.cfgs.type)
            self.triplane = Triplane(config=self.cfgs.triplane, data_type=self.data_type, device=device,
                                     model_type=self.cfgs.type)
            self.e_mu = nn.Linear(self.cfgs.cvae_in_dim, self.cfgs.cvae_in_dim)
            self.e_logvar = nn.Linear(self.cfgs.cvae_in_dim, self.cfgs.cvae_in_dim)
            self.decoder = nn.Sequential(
                nn.Linear(self.cfgs.triplane.pe_num_fts, self.cfgs.triplane.hidden_dims), nn.Softplus(),
                nn.Linear(self.cfgs.triplane.hidden_dims, self.cfgs.triplane.n_classes * 8)
            )
            hierarchical_init(self.decoder)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch):
        for key in batch.keys():
            if key != DataDict.name:
                batch[key] = batch[key].detach().cpu()

        if self.model_type == ModelTypes.triplane:
            vox_feats, img_feats = self.baseline(batch)
            voxel = self.triplane(vox_feats=vox_feats, img_feats=img_feats, proposal=batch['proposal'].numpy(),
                                  projection_matrix=batch[DataDict.w2i], pc_range=batch[DataDict.pc_range])
            sparse_vox = self.decoder(voxel)
            vox = rearrange(sparse_vox, "b x y z (c d e f) -> b (x d) (y e) (z f) c", d=2, e=2, f=2)
            return {DataDict.ssc_pred: vox.permute(0, 4, 1, 2, 3)}
        elif self.model_type == ModelTypes.cvae:
            vox_feats, img_feats = self.baseline(batch)

            mu = self.e_mu(vox_feats)
            logvar = self.e_logvar(vox_feats)
            if self.training:
                z = self._reparameterize(mu, logvar)
            else:
                z = mu
            voxel = self.triplane(vox_feats=z, img_feats=img_feats, proposal=batch['proposal'].numpy(),
                                  projection_matrix=batch[DataDict.w2i], pc_range=batch[DataDict.pc_range])
            sparse_vox = self.decoder(voxel)
            vox = rearrange(sparse_vox, "b x y z (c d e f) -> b (x d) (y e) (z f) c", d=2, e=2, f=2)
            return {DataDict.mu: mu, DataDict.log_var: logvar, DataDict.ssc_pred: vox.permute(0, 4, 1, 2, 3)}
        elif self.model_type == ModelTypes.query:
            voxels = self.baseline(batch)
            pred = self.output(voxels).permute(0, 4, 1, 2, 3)
            return {DataDict.ssc_pred: pred}

