import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from loss_metrics.functions import compute_super_CP_multilabel_loss, CE_ssc_loss, sem_scal_loss, geo_scal_loss, KL_sep, \
    BCE_ssc_loss
from loss_metrics.sscMetrics import SSCMetrics
from utilities.configs import ModelTypes, LossNames, DataDict, EvaluationNames
from utilities.functions import to_device


class Loss(nn.Module):
    def __init__(self, config, device, datatype, model_type):
        super().__init__()
        self.model_type = model_type
        self.device = device
        self.data_type = datatype
        self.cfg = config
        self.class_weights_level_1 = torch.from_numpy(1 / np.log(self.cfg.class_frequencies_level1 + 0.001))
        self.alpha = config.alpha
        if self.model_type == ModelTypes.query:
            only_iou = True
        else:
            only_iou = False
        self.metric = SSCMetrics(config.n_classes, only_iou=only_iou)

    def forward(self, output_dict, input_dict):
        loss_dict = dict()
        ssc_pred = output_dict[DataDict.ssc_pred]
        target = to_device(input_dict[DataDict.target], device=self.device)

        if self.model_type == ModelTypes.query:
            class_weights_level_1 = to_device(self.class_weights_level_1.to(torch.float32), device=self.device)
            loss = BCE_ssc_loss(ssc_pred, target, class_weights_level_1, self.alpha)
        else:
            class_weight = to_device(torch.from_numpy(self.cfg.class_weights).type_as(input_dict[DataDict.img]),
                                     device=self.device) * 10.0
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight, use_mean=self.cfg.ssc_mean)
            loss_dict[LossNames.ce_ssc_loss] = loss_ssc

            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            loss_dict[LossNames.sem_scal] = loss_sem_scal

            if self.cfg.use_occ:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict[LossNames.geo_scal] = loss_geo_scal
            else:
                loss_geo_scal = 0

            loss = loss_ssc + loss_sem_scal + loss_geo_scal

            if self.model_type == ModelTypes.cvae:
                logvar = output_dict[DataDict.log_var]
                mu = output_dict[DataDict.mu]
                cvae_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss_dict[LossNames.cvae] = cvae_loss
                loss += cvae_loss * self.cfg.cvae_ratio
        loss_dict[LossNames.loss] = loss
        return loss_dict

    def evaluate(self, output_dict, input_dict):
        y_pred = np.argmax(output_dict[DataDict.ssc_pred].detach().cpu().numpy(), axis=1)

        self.metric.add_batch(y_pred=y_pred, y_true=input_dict[DataDict.target].detach().cpu().numpy())
        results = self.metric.get_stats()
        results_dict = {}
        if EvaluationNames.iou_ssc_mean in results.keys():
            for i, class_name in enumerate(self.cfg.class_names):
                results_dict["{}/{}".format(EvaluationNames.iou_ssc, class_name)] = results[EvaluationNames.iou_ssc][i]
            results_dict[EvaluationNames.iou_ssc_mean] = results[EvaluationNames.iou_ssc_mean]
        results_dict[EvaluationNames.iou_occ] = results[EvaluationNames.iou_occ]
        results_dict[EvaluationNames.precision] = results[EvaluationNames.precision]
        results_dict[EvaluationNames.recall] = results[EvaluationNames.recall]
        return results_dict
