# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from data.semantic_kitti.pre_process import KITTIProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data_root', type=str, help='root of the dataset', default="")
    parser.add_argument('--render', action='store_true', default=False, help='if to display for debug')
    parser.add_argument('--batch', type=int, default=1, help="batch of processing")
    parser.add_argument('--index', type=int, default=0, help="index of the batch")
    parser.add_argument('--type', type=int, default=0, help="0: generate target; 1: generate raw proposed queries; 2: generate our queries")
    parser.add_argument('--data_config', type=str, help='configuration file',
                        default="data/semantic_kitti/semantic-kitti.yaml")
    args = parser.parse_args()

    processor = KITTIProcessor(args.data_config, args.data_root, args.render)
    processor.run(args.batch, args.index)


