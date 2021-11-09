from time import perf_counter
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.compose import Compose


@PIPELINES.register_module()
class Timer(Compose):
    """Times a list of transforms and stores result in img_meta."""

    def __init__(self, name, transforms):
        super(Timer, self).__init__(transforms)
        self.name = f"{name}_time"

    def __call__(self, data):
        t1 = perf_counter()
        data = super(Timer, self).__call__(data)
        data[self.name] = perf_counter() - t1
        return data


@PIPELINES.register_module()
class DummyResize(object):
    """Replacement for resize in case the scale is 1.
    Adds img_shape, pad_shape, scale_factor to results."""

    def __call__(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            img_shape = results[key].shape
            results['img_shape'] = img_shape
            # in case that there is no padding
            results['pad_shape'] = img_shape
            results['scale_factor'] = np.array([1, 1, 1, 1], dtype=np.float32)
        return results


@PIPELINES.register_module()
class ImageTestTransformGPU(object):
    """Preprocess an image using GPU."""

    def __init__(self, img_norm_cfg, size_divisor, scale_factor):
        self.img_norm_cfg = img_norm_cfg
        self.mean = torch.tensor(img_norm_cfg['mean'], dtype=torch.float32)
        self.std = torch.tensor(img_norm_cfg['std'], dtype=torch.float32)
        self.to_rgb = img_norm_cfg['to_rgb']
        self.std_inv = 1/self.std
        self.size_divisor = size_divisor
        self.scale_factor = float(scale_factor)

    def __call__(self, results, device='cuda'):
        start = perf_counter()
        img = results['img']
        ori_shape = img.shape
        h, w = img.shape[:2]
        new_size = (round(h*self.scale_factor), round(w*self.scale_factor))
        img_shape = (*new_size, 3)

        img = torch.from_numpy(img).to(device).float()
        if self.to_rgb:
            img = img[:, :, (2, 1, 0)]
        # to BxCxHxW
        img = img.permute(2, 0, 1).unsqueeze(0)

        if new_size[0] != img.shape[2] or new_size[1] != img.shape[3]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ignore the align_corner warnings
                img = F.interpolate(img, new_size, mode='bilinear')

        for c in range(3):
            img[:, c, :, :] = img[:, c, :, :].sub(self.mean[c]) \
                                             .mul(self.std_inv[c])

        if self.size_divisor is not None:
            pad_h = int(np.ceil(new_size[0] / self.size_divisor)) \
                * self.size_divisor - new_size[0]
            pad_w = int(np.ceil(new_size[1] / self.size_divisor)) \
                * self.size_divisor - new_size[1]
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            pad_shape = (img.shape[2], img.shape[3], 3)
        else:
            pad_shape = img_shape

        img_meta = dict(
            filename=results['filename'],
            ori_filename=results['ori_filename'],
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=self.scale_factor,
            flip=False,
            img_norm_cfg=self.img_norm_cfg,
            start_time=start,
        )

        if 'gt_bboxes' in results:
            gt_bboxes = torch.from_numpy(results['gt_bboxes']) \
                             .to(device).float()
            gt_labels = torch.from_numpy(results['gt_labels']) \
                             .to(device).float()
            return dict(img=img, img_metas=[img_meta],
                        gt_bboxes=gt_bboxes, gt_labels=gt_labels)
        else:
            return dict(img=img, img_metas=[img_meta])
