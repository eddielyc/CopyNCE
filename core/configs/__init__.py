# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import pathlib

from omegaconf import OmegaConf

__all__ = [
    "load_config",
    "default_config",
    "matching_default_config",
    "descriptor_default_config",
    "__default_config_factory__",
]


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


default_config = load_config("matching_default_config")
matching_default_config = load_config("matching_default_config")
descriptor_default_config = load_config("descriptor_default_config")


__default_config_factory__ = {
    "default": default_config,
    "matching": matching_default_config,
    "descriptor": descriptor_default_config,
    "cnn_descriptor": descriptor_default_config,
}
