import numpy as np
import os

from mmdet.models.builder import DETECTORS, build
from mmcv.runner import load_checkpoint


def mkdir2(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def print_stats(var, name='', fmt='%.3g', cvt=lambda x: x):
    var = np.asarray(var)
    if name:
        prefix = name + ': '
    else:
        prefix = ''
    if len(var) == 1:
        print(('%sscalar: ' + fmt) % (
            prefix,
            cvt(var[0]),
        ))
    else:
        fmt_str = 'mean: %s; std: %s; min: %s; max: %s' % (
            fmt, fmt, fmt, fmt
        )
        print(('%s' + fmt_str) % (
            prefix,
            cvt(var.mean()),
            cvt(var.std(ddof=1)),
            cvt(var.min()),
            cvt(var.max()),
        ))


def apply_class_map(result, class_mapping, num_new_classes):
    """Maps result via the class mapping.

    Args:
        result (list[any]): list of class-specific results of length equal \
            to the number of original classes
        class_mapping (dict[int, int]): maps index of original classes to \
            index of new classes
        num_new_classes (int): number of new classes
    """
    mapped = [np.empty((0, 5)) for c in range(num_new_classes)]
    for i1, i2 in class_mapping.items():
        mapped[i2] = result[i1]
    return mapped


def digit_version(version_str):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions.

    Args:
        version_str (str): The version string.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return tuple(digit_version)


def get_mmdet_hash():
    """Get the git hash of the installed mmdetection repo."""
    return '44a7ef2e80f355defb943d02bbee4a011b362a9d'


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector.
    Modification: If cfg has load_from, then the outputted detector will load \
    weights from the given file."""
    if hasattr(cfg, 'load_from'):
        load_from = cfg.load_from
        delattr(cfg, 'load_from')
    else:
        load_from = None

    detector = build(cfg, DETECTORS,
                     dict(train_cfg=train_cfg, test_cfg=test_cfg))

    if load_from is not None:
        load_checkpoint(detector, load_from)

    return detector
