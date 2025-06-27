# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from loss_metrics.loss import Loss
from utilities.configs import TrainingConfig, EvaluationNames
from utilities.common import BasePipeline
from data.data_loader import evaluation_data_loader, train_data_loader
from utilities.arguments import get_configuration
from utilities.kitti_configs import DataDict


class Evaluator(BasePipeline):
    def __init__(self, cfgs: TrainingConfig):
        """
        This class is the evaluation
        Args:
            cfgs: the configuration of the training class
        """
        super().__init__(cfgs)
        self.cfgs = cfgs
        self.generate_dataset = cfgs.generate_data
        self.evaluation_data_loader = evaluation_data_loader(cfg=cfgs.data, model_type=cfgs.model.type)
        if self.generate_dataset:
            self.output_dir = os.path.join(cfgs.data.data_root, "dataset/sequences_queries")
            self.training_data_loader = train_data_loader(cfg=cfgs.data, model_type=cfgs.model.baseline_type)

        self.set_eval_mode()

        self.model_type = self.cfgs.model.type

        if self.device == "cuda":
            self.loss_func = Loss(config=cfgs.loss, device=self.device, datatype=cfgs.data.data_type,
                                  model_type=cfgs.model.type).cuda()
        else:
            self.loss_func = Loss(config=cfgs.loss, device=self.device, datatype=cfgs.data.data_type,
                                  model_type=cfgs.model.type).to(self.device)

    def cleanup(self):
        dist.destroy_process_group()

    def step(self, data_dict) -> Tuple[dict, dict]:
        """
        one step of the model, loss function and also the loss_metrics
        Args:
            train:
            data_dict: the input data dictionary
        Returns:
            the output from the model, the output from the loss function
        """
        # start_time = time.time()
        output_dict = self.model(data_dict)
        torch.cuda.empty_cache()
        return output_dict

    def run_query(self):
        loss_dict = {}
        for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader, desc="Evaluation")):
            output_dict = self.step(data_dict)
            loss_dict = self.loss_func.evaluate(output_dict=output_dict, input_dict=data_dict)
        print(loss_dict)

    def calculate_single_iou(self, y_pred, y_true, only_iou=False):
        mask = y_true != 255
        tp, fp, fn = self.loss_func.metric.get_score_completion(y_pred, y_true, mask)
        results = {
            EvaluationNames.precision: tp / (tp + fp),
            EvaluationNames.iou_occ: tp / (tp + fp + fn),
            EvaluationNames.recall: tp / (tp + fn)
        }
        if not only_iou:
            mask = y_true != 255
            tp_sum, fp_sum, fn_sum = self.loss_func.metric.get_score_semantic_and_completion(y_pred, y_true, mask)
            iou_ssc = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-5)
            results.update({EvaluationNames.iou_ssc: iou_ssc})
        return results

    def run(self, render=False):
        """
        run the evaluation process
        """
        if self.generate_dataset:
            self.run_data_generation()
        else:
            torch.autograd.set_detect_anomaly(True)
            all_metrics = []
            for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader, desc="Evaluation")):
                output_dict = self.step(data_dict)
                loss_dict = self.loss_func.evaluate(output_dict=output_dict, input_dict=data_dict)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            print(loss_dict)

    def run_data_generation(self):
        for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader, desc="Evaluation dataset")):
            output_dict = self.step(data_dict)
            y_pred = np.argmax(output_dict[DataDict.ssc_pred].detach().cpu().numpy(), axis=1)[0].flatten()
            sequence_folder = os.path.join(self.output_dir, data_dict[DataDict.name][0][:2])
            if not os.path.exists(sequence_folder):
                os.makedirs(sequence_folder)
            output_path = os.path.join(self.output_dir, data_dict[DataDict.name][0] + ".pkl")
            with open(output_path, "wb") as handle:
                pickle.dump(y_pred, handle)
            torch.cuda.empty_cache()

            loss_dict = self.loss_func.evaluate(output_dict=output_dict, input_dict=data_dict)
        print(loss_dict)
        for iteration, data_dict in enumerate(tqdm(self.training_data_loader, desc="Training dataset")):
            output_dict = self.step(data_dict)
            y_pred = np.argmax(output_dict[DataDict.ssc_pred].detach().cpu().numpy(), axis=1)[0].flatten()
            sequence_folder = os.path.join(self.output_dir, data_dict[DataDict.name][0][:2])
            if not os.path.exists(sequence_folder):
                os.makedirs(sequence_folder)
            output_path = os.path.join(self.output_dir, data_dict[DataDict.name][0] + ".pkl")
            with open(output_path, "wb") as handle:
                pickle.dump(y_pred, handle)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    cfgs = get_configuration()
    trainer = Evaluator(cfgs=cfgs)
    torch.autograd.set_detect_anomaly(True)
    trainer.run(render=False)
    torch.autograd.set_detect_anomaly(False)


