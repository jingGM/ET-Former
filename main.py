# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from train import Trainer
from utilities.arguments import get_configuration


if __name__ == "__main__":
    cfgs = get_configuration()
    trainer = Trainer(cfgs=cfgs)
    torch.autograd.set_detect_anomaly(True)
    trainer.run()
    torch.autograd.set_detect_anomaly(False)
