# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Optional, List
import argparse

from core import project_root
from core.train.parallel_train import main as main_train


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = tuple(),
    add_help: bool = True,
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--arch",
        default="matching",
        type=str,
        choices=["matching", "descriptor", "cnn_descriptor"],
        help="Arch of train model, must be one of ['matching', 'descriptor', 'cnn_descriptor']",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        type=str,
        nargs="*",
        help="path to config file"
    )
    parser.add_argument(
        "opts",
        help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs.
    For python-based LazyConfig, use "path.key=value".
            """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        nargs="*",
        help="Output directory to save logs and checkpoints",
    )

    return parser


def main():
    description = "Launch image copy detection training."
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    assert all(os.path.exists(cfg_file) for cfg_file in args.config_file), "Configuration file does not exist!"
    main_train(args)
    return 0


if __name__ == "__main__":
    main()
