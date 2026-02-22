# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import sys
from pathlib import Path
from loguru import logger

from .helpers import MetricLogger, SmoothedValue


def setup_logging(output_dir=None, incremental=False, cmd=True):
    rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    if not incremental:
        logger.remove(handler_id=None)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.add(output_dir / f"rank-{rank}.log", enqueue=True)

    if rank == 0 and cmd:
        logger.add(sys.stdout, level="DEBUG")
