from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet.models.detectors import BaseDetector

from ..utils import build_detector
from ..utils.vis import overlay_grid, vis_batched_imgs
from .grid_generator import build_grid_generator


@DETECTORS.register_module()
class FOVEAWarp(BaseDetector):
    """Wrapper class for enhancing a detector with attentional warping."""

    def __init__(self,
                 task_detector,
                 grid_net,
                 pretrained=None,  # for compatibility with mmdet tools
                 vis_options={},
                 train_cfg=None,
                 test_cfg=None):
        super(FOVEAWarp, self).__init__()
        self.task_detector = build_detector(
            task_detector, train_cfg=train_cfg, test_cfg=test_cfg)
        self.grid_net = build_grid_generator(grid_net)
        self.vis_options = vis_options
        self.count = 0

    def extract_feat(self, img):
        pass

    def _get_scale_factor(self, ori_shape, new_shape):
        ori_height, ori_width, _ = ori_shape
        img_height, img_width, _ = new_shape
        w_scale = img_width / ori_width
        h_scale = img_height / ori_height
        return np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        self.count += 1
        if 'input_image' in self.vis_options:
            vis_batched_imgs(self.vis_options['input_image'], imgs,
                             img_metas, index=self.count)
        grid = self.grid_net(imgs, img_metas, jitter=50, **kwargs)
        warped_imgs = F.grid_sample(imgs, grid, align_corners=True)
        if 'warped_image' in self.vis_options:
            vis_batched_imgs(self.vis_options['warped_image'], warped_imgs,
                             img_metas, index=self.count)
        # update img metas, assuming that all imgs have the same original shape
        img_height, img_width = grid.shape[1:3]
        img_shape = (img_height, img_width, 3)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = self._get_scale_factor(ori_shape, img_shape)
        # TODO: undo hardcoding size divisor of 32
        pad_h = int(np.ceil(img_shape[0] / 32)) * 32 - img_shape[0]
        pad_w = int(np.ceil(img_shape[1] / 32)) * 32 - img_shape[1]
        warped_imgs = F.pad(
            warped_imgs, (0, pad_w, 0, pad_h), mode='constant', value=0)
        pad_shape = (warped_imgs.shape[2], warped_imgs.shape[3], 3)
        for i in range(len(img_metas)):
            img_metas[i]['warp_grid'] = grid[i]
            img_metas[i]['img_shape'] = img_shape
            img_metas[i]['scale_factor'] = scale_factor
            img_metas[i]['pad_shape'] = pad_shape
            img_metas[i]['pad_fixed_size'] = None
            img_metas[i]['pad_size_divisor'] = 32
        losses = self.task_detector.forward_train(
            warped_imgs, img_metas, **kwargs)
        return losses

    def simple_test(self, imgs, img_metas, rescale=False, **kwargs):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        vis_options = kwargs.get('vis_options', self.vis_options)
        if 'vis_options' in kwargs:
            vis_options = kwargs.pop('vis_options')

        start = perf_counter()
        imgs = torch.stack(tuple(imgs), dim=0)
        if 'input_image' in vis_options:
            show_imgs = imgs.clone()
            vis_batched_imgs(vis_options['input_image']+'_no_box_no_grid',
                             show_imgs, img_metas, bboxes=None)
            show_imgs = overlay_grid(show_imgs)  # overlay a red grid
            vis_batched_imgs(vis_options['input_image'], show_imgs, img_metas,
                             bboxes=kwargs.get('gt_bboxes', None))
            vis_batched_imgs(vis_options['input_image']+'_no_box', show_imgs,
                             img_metas, bboxes=None)

        grid = self.grid_net(imgs, img_metas, **kwargs)
        if 'gt_bboxes' in kwargs:
            del kwargs['gt_bboxes']
        if 'gt_labels' in kwargs:
            del kwargs['gt_labels']

        if 'vis_options' in kwargs:
            vis_options = kwargs.pop('vis_options')

        if 'magnification_heatmap' in vis_options:
            import matplotlib as mpl
            import matplotlib.cm as cm
            x = (grid[:, :, :, 0]+1)/2*grid.shape[2]
            y = (grid[:, :, :, 1]+1)/2*grid.shape[1]
            x_ratio = 1/(x[:, :-1, 1:]-x[:, :-1, :-1])
            y_ratio = 1/(y[:, 1:, :-1]-y[:, :-1, :-1])
            magnification_ratios = (x_ratio * y_ratio) ** 0.5
            norm = mpl.colors.Normalize(vmin=0.5, vmax=2.0)
            cmap = cm.viridis
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            heatmap = torch.stack([
                torch.Tensor(m.to_rgba(magnification_ratio))
                for magnification_ratio in magnification_ratios.detach()
                                                               .cpu().numpy()
            ])
            heatmap = (1-heatmap[:, :, :, 3:]) + heatmap[:, :, :, :3] \
                * heatmap[:, :, :, 3:]
            heatmap = 255 * heatmap.permute(0, 3, 1, 2)
            heatmap = heatmap[:, (2, 1, 0), :, :]
            vis_batched_imgs(vis_options['magnification_heatmap'], heatmap,
                             img_metas, bboxes=None, denorm=False)

        warped_imgs = F.grid_sample(imgs, grid, align_corners=True)

        # update img metas, assuming that all imgs have the same original shape
        warp_time = perf_counter() - start
        img_height, img_width = grid.shape[1:3]
        img_shape = (img_height, img_width, 3)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = self._get_scale_factor(ori_shape, img_shape)
        # TODO: undo hardcoding size divisor of 32
        pad_h = int(np.ceil(img_shape[0] / 32)) * 32 - img_shape[0]
        pad_w = int(np.ceil(img_shape[1] / 32)) * 32 - img_shape[1]
        warped_imgs = F.pad(
            warped_imgs, (0, pad_w, 0, pad_h), mode='constant', value=0)
        pad_shape = (warped_imgs.shape[2], warped_imgs.shape[3], 3)
        for i in range(len(img_metas)):
            img_metas[i]['warp_grid'] = grid[i]
            img_metas[i]['warp_time'] = warp_time
            img_metas[i]['img_shape'] = img_shape
            img_metas[i]['scale_factor'] = scale_factor
            img_metas[i]['pad_shape'] = pad_shape
            img_metas[i]['pad_fixed_size'] = None
            img_metas[i]['pad_size_divisor'] = 32

        results = self.task_detector.simple_test(
            warped_imgs, img_metas, rescale=rescale,
            store_warped_det=('warped_image' in vis_options), **kwargs)

        if 'warped_image' in vis_options:
            thresholded_bboxes = []
            for i in range(len(img_metas)):
                bbox = img_metas[i]['warped_detections']
                if bbox.shape[1] > 4:  # includes confidences
                    bbox = bbox[bbox[:, 4] > 0.5, :4]
                thresholded_bboxes.append(bbox)
            show_warped_imgs = F.grid_sample(show_imgs, grid,
                                             align_corners=True)
            labels = [
                img_meta.get('warped_labels', None)
                for img_meta in img_metas
            ]
            vis_batched_imgs(
                vis_options['warped_image']+'_thresh', show_warped_imgs,
                img_metas, bboxes=thresholded_bboxes, labels=labels)
            vis_batched_imgs(
                vis_options['warped_image']+'_thresh_no_grid', warped_imgs,
                img_metas, bboxes=thresholded_bboxes, labels=labels)
            vis_batched_imgs(
                vis_options['warped_image']+'_thresh_no_box_no_grid',
                warped_imgs, img_metas, bboxes=None, labels=labels)

        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        pass
