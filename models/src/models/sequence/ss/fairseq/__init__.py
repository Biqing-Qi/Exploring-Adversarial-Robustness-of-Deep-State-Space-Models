# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ['pdb']
__version__ = '0.9.0'

import sys

# backwards compatibility to support `from fairseq.meters import AverageMeter`
from .logging import meters, metrics, progress_bar  # noqa
sys.modules['fairseq.meters'] = meters
sys.modules['fairseq.metrics'] = metrics
sys.modules['fairseq.progress_bar'] = progress_bar

from . import models  # noqa
from . import modules  # noqa
from . import pdb  # noqa
