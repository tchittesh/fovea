import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import Registry
from mmdet.models.builder import build

from ..utils.vis import vis_batched_imgs

GRID_GENERATORS = Registry('grid_generator')


def build_grid_generator(cfg):
    """Build grid generator."""
    return build(cfg, GRID_GENERATORS)


def make1DGaussian(size, fwhm=3, center=None):
    """ Make a 1D gaussian kernel.

    size is the length of the kernel,
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, dtype=np.float)

    if center is None:
        center = size // 2

    return np.exp(-4*np.log(2) * (x-center)**2 / fwhm**2)


def make2DGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def unwarp_bboxes(bboxes, grid, output_shape):
    """Unwarps a tensor of bboxes of shape (n, 4) or (n, 5) according to the grid \
    of shape (h, w, 2) used to warp the corresponding image and the \
    output_shape (H, W, ...)."""
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(2, 0, 1).unsqueeze(0)

    warped_height, warped_width = grid.shape[0:2]
    xgrid = 2 * (bboxes[:, 0:4:2] / warped_width) - 1
    ygrid = 2 * (bboxes[:, 1:4:2] / warped_height) - 1
    grid = torch.stack((xgrid, ygrid), dim=2).unsqueeze(0)

    # warped_bboxes has shape (2, num_bboxes, 2)
    warped_bboxes = F.grid_sample(
        img, grid, align_corners=True, padding_mode="border").squeeze(0)
    bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * output_shape[1]
    bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * output_shape[0]

    return bboxes


class RecasensSaliencyToGridMixin(object):
    """Grid generator based on 'Learning to Zoom: a Saliency-Based Sampling \
    Layer for Neural Networks' [https://arxiv.org/pdf/1809.03355.pdf]."""

    def __init__(self, output_shape, grid_shape=(31, 51), separable=True,
                 attraction_fwhm=13, anti_crop=True, **kwargs):
        super(RecasensSaliencyToGridMixin, self).__init__()
        self.output_shape = output_shape
        self.output_height, self.output_width = output_shape
        self.grid_shape = grid_shape
        self.padding_size = min(self.grid_shape)-1
        self.total_shape = tuple(
            dim+2*self.padding_size
            for dim in self.grid_shape
        )
        self.padding_mode = 'reflect' if anti_crop else 'replicate'
        self.separable = separable

        if self.separable:
            self.filter = make1DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter).unsqueeze(0) \
                                                        .unsqueeze(0).cuda()

            self.P_basis_x = torch.zeros(self.total_shape[1])
            for i in range(self.total_shape[1]):
                self.P_basis_x[i] = \
                    (i-self.padding_size)/(self.grid_shape[1]-1.0)
            self.P_basis_y = torch.zeros(self.total_shape[0])
            for i in range(self.total_shape[0]):
                self.P_basis_y[i] = \
                    (i-self.padding_size)/(self.grid_shape[0]-1.0)
        else:
            self.filter = make2DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter) \
                               .unsqueeze(0).unsqueeze(0).cuda()

            self.P_basis = torch.zeros(2, *self.total_shape)
            for k in range(2):
                for i in range(self.total_shape[0]):
                    for j in range(self.total_shape[1]):
                        self.P_basis[k, i, j] = k*(i-self.padding_size)/(self.grid_shape[0]-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_shape[1]-1.0)  # noqa: E501

    def separable_saliency_to_grid(self, imgs, img_metas, x_saliency,
                                   y_saliency, device):
        assert self.separable
        x_saliency = F.pad(x_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)
        y_saliency = F.pad(y_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)

        N = imgs.shape[0]
        P_x = torch.zeros(1, 1, self.total_shape[1], device=device)
        P_x[0, 0, :] = self.P_basis_x
        P_x = P_x.expand(N, 1, self.total_shape[1])
        P_y = torch.zeros(1, 1, self.total_shape[0], device=device)
        P_y[0, 0, :] = self.P_basis_y
        P_y = P_y.expand(N, 1, self.total_shape[0])

        weights = F.conv1d(x_saliency, self.filter)
        weighted_offsets = torch.mul(P_x, x_saliency)
        weighted_offsets = F.conv1d(weighted_offsets, self.filter)
        xgrid = weighted_offsets/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, 1, self.grid_shape[1])
        xgrid = xgrid.expand(-1, 1, *self.grid_shape)

        weights = F.conv1d(y_saliency, self.filter)
        weighted_offsets = F.conv1d(torch.mul(P_y, y_saliency), self.filter)
        ygrid = weighted_offsets/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, self.grid_shape[0], 1)
        ygrid = ygrid.expand(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        grid = F.interpolate(grid, size=self.output_shape, mode='bilinear',
                             align_corners=True)
        return grid.permute(0, 2, 3, 1)

    def nonseparable_saliency_to_grid(self, imgs, img_metas, saliency, device):
        assert not self.separable
        p = self.padding_size
        saliency = F.pad(saliency, (p, p, p, p), mode=self.padding_mode)

        N = imgs.shape[0]
        P = torch.zeros(1, 2, *self.total_shape, device=device)
        P[0, :, :, :] = self.P_basis
        P = P.expand(N, 2, *self.total_shape)

        saliency_cat = torch.cat((saliency, saliency), 1)
        weights = F.conv2d(saliency, self.filter)
        weighted_offsets = torch.mul(P, saliency_cat) \
                                .view(-1, 1, *self.total_shape)
        weighted_offsets = F.conv2d(weighted_offsets, self.filter) \
                            .view(-1, 2, *self.grid_shape)

        weighted_offsets_x = weighted_offsets[:, 0, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        xgrid = weighted_offsets_x/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, *self.grid_shape)

        weighted_offsets_y = weighted_offsets[:, 1, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        ygrid = weighted_offsets_y/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        grid = F.interpolate(grid, size=self.output_shape, mode='bilinear',
                             align_corners=True)
        return grid.permute(0, 2, 3, 1)


@GRID_GENERATORS.register_module()
class PlainKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """Image adaptive grid generator with fixed hyperparameters -- KDE SI"""

    def __init__(
        self,
        bandwidth_scale=1,
        amplitude_scale=1,
        **kwargs
    ):
        super(PlainKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.bandwidth_scale = bandwidth_scale
        self.amplitude_scale = amplitude_scale

    def bbox2sal(self, batch_bboxes, img_metas, jitter=None):
        device = batch_bboxes[0].device
        h_out, w_out = self.grid_shape
        sals = []
        for i in range(len(img_metas)):
            h, w, _ = img_metas[i]['pad_shape']
            bboxes = batch_bboxes[i]
            if len(bboxes) == 0:  # zero detections case
                sal = torch.ones(h_out, w_out, device=device).unsqueeze(0)
                sal /= sal.sum()
                sals.append(sal)
                continue
            bboxes[:, 2:] -= bboxes[:, :2]  # ltrb -> ltwh
            cxy = bboxes[:, :2] + 0.5*bboxes[:, 2:]
            if jitter is not None:
                cxy += 2*jitter*(torch.randn(cxy.shape, device=device)-0.5)

            widths = (bboxes[:, 2] * self.bandwidth_scale).unsqueeze(1)
            heights = (bboxes[:, 3] * self.bandwidth_scale).unsqueeze(1)

            X, Y = torch.meshgrid(
                torch.linspace(0, w, w_out, dtype=torch.float, device=device),
                torch.linspace(0, h, h_out, dtype=torch.float, device=device),
            )
            grids = torch.stack((X.flatten(), Y.flatten()), dim=1).t()

            m, n = cxy.shape[0], grids.shape[1]

            norm1 = (cxy[:, 0:1]**2/widths + cxy[:, 1:2]**2/heights) \
                .expand(m, n)
            norm2 = grids[0:1, :]**2/widths + grids[1:2, :]**2/heights
            norms = norm1 + norm2

            cxy_norm = cxy
            cxy_norm[:, 0:1] /= widths
            cxy_norm[:, 1:2] /= heights

            distances = norms - 2*cxy_norm.mm(grids)

            sal = (-0.5 * distances).exp()
            sal = self.amplitude_scale * (sal / (0.00001+sal.sum(dim=1, keepdim=True)))  # noqa: E501, normalize each distribution
            sal += 1/((2*self.padding_size+1)**2)
            sal = sal.sum(dim=0)
            sal /= sal.sum()
            sal = sal.reshape(w_out, h_out).t().unsqueeze(0)  # noqa: E501, add channel dimension
            sals.append(sal)
        return torch.stack(sals)

    def forward(self, imgs, img_metas, gt_bboxes, jitter=False, **kwargs):
        vis_options = kwargs.get('vis_options', {})

        if isinstance(gt_bboxes, torch.Tensor):
            batch_bboxes = gt_bboxes
        else:
            if len(gt_bboxes[0].shape) == 3:
                batch_bboxes = gt_bboxes[0].clone()  # noqa: E501, removing the augmentation dimension
            else:
                batch_bboxes = [bboxes.clone() for bboxes in gt_bboxes]
        device = batch_bboxes[0].device
        saliency = self.bbox2sal(batch_bboxes, img_metas, jitter)

        if 'saliency' in vis_options:
            h, w, _ = img_metas[0]['pad_shape']
            show_saliency = F.interpolate(saliency, size=(h, w),
                                          mode='bilinear', align_corners=True)
            show_saliency = 255*(show_saliency/show_saliency.max())
            show_saliency = show_saliency.expand(
                show_saliency.size(0), 3, h, w)
            vis_batched_imgs(vis_options['saliency'], show_saliency,
                             img_metas, bboxes=gt_bboxes, denorm=False)
            vis_batched_imgs(vis_options['saliency']+'_no_box', show_saliency,
                             img_metas, bboxes=None, denorm=False)

        if self.separable:
            x_saliency = saliency.sum(dim=2)
            y_saliency = saliency.sum(dim=3)
            grid = self.separable_saliency_to_grid(imgs, img_metas, x_saliency,
                                                   y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs, img_metas,
                                                      saliency, device)

        return grid


@GRID_GENERATORS.register_module()
class LearnedKDEGrid(PlainKDEGrid):
    """Image adaptive grid generator with learned hyperparameters -- LKDE SI"""

    def __init__(self, bandwidth_init, amplitude_init, **kwargs):
        PlainKDEGrid.__init__(self, **kwargs)
        self.bandwidth_init = bandwidth_init
        self.amplitude_init = amplitude_init
        self.bandwidth = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.amplitude = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def bbox2sal(self, batch_bboxes, img_metas, jitter):
        self.bandwidth_scale = \
            ((1+self.bandwidth).abs() * self.bandwidth_init) + 0.1
        self.amplitude_scale = \
            ((1+self.amplitude).abs() * self.amplitude_init) + 0.1
        return PlainKDEGrid.bbox2sal(self, batch_bboxes, img_metas, jitter)


@GRID_GENERATORS.register_module()
class FixedKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """Grid generator that uses a fixed saliency map -- KDE SD"""

    def __init__(self, saliency_file, **kwargs):
        super(FixedKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.saliency = pickle.load(open(saliency_file, 'rb'))

    def forward(self, imgs, img_metas, **kwargs):
        vis_options = kwargs.get('vis_options', {})
        device = imgs.device
        self.saliency = self.saliency.to(device)

        if 'saliency' in vis_options:
            h, w, _ = img_metas[0]['pad_shape']
            show_saliency = F.interpolate(self.saliency, size=(h, w),
                                          mode='bilinear', align_corners=True)
            show_saliency = 255*(show_saliency/show_saliency.max())
            show_saliency = show_saliency.expand(
                show_saliency.size(0), 3, h, w)
            vis_batched_imgs(vis_options['saliency'], show_saliency,
                             img_metas, denorm=False)
            vis_batched_imgs(vis_options['saliency']+'_no_box', show_saliency,
                             img_metas, bboxes=None, denorm=False)

        if self.separable:
            x_saliency = self.saliency.sum(dim=2)
            y_saliency = self.saliency.sum(dim=3)
            grid = self.separable_saliency_to_grid(imgs, img_metas, x_saliency,
                                                   y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs, img_metas,
                                                      self.saliency, device)

        return grid


@GRID_GENERATORS.register_module()
class CompositeKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """Grid generator that uses a weighted combination of a fixed and an
    adaptive saliency maps -- KDE SC alpha is the weight given to the
    adaptive_grid."""

    def __init__(self, fixed_grid, adaptive_grid, alpha, **kwargs):
        super(CompositeKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        assert 0 <= alpha <= 1
        self.alpha = alpha
        fixed_grid.update(**kwargs)
        adaptive_grid.update(**kwargs)
        self.fixed_grid = build_grid_generator(fixed_grid)
        self.adaptive_grid = build_grid_generator(adaptive_grid)
        assert self.separable == self.fixed_grid.separable and \
            self.separable == self.adaptive_grid.separable

    def forward(self, imgs, img_metas, gt_bboxes, jitter=False, **kwargs):
        vis_options = kwargs.get('vis_options', {})
        device = imgs.device
        if len(gt_bboxes[0].shape) == 3:
            batch_bboxes = gt_bboxes[0].clone()  # noqa: E501, removing the augmentation dimension
        else:
            batch_bboxes = [bboxes.clone() for bboxes in gt_bboxes]
        adaptive_saliency = self.adaptive_grid.bbox2sal(
            batch_bboxes, img_metas, jitter)
        saliency = self.fixed_grid.saliency.to(device) * (1-self.alpha) + \
            adaptive_saliency * self.alpha

        if 'saliency' in vis_options:
            h, w, _ = img_metas[0]['pad_shape']
            show_saliency = F.interpolate(saliency, size=(h, w),
                                          mode='bilinear', align_corners=True)
            show_saliency = 255*(show_saliency/show_saliency.max())
            show_saliency = show_saliency.expand(
                show_saliency.size(0), 3, h, w)
            vis_batched_imgs(vis_options['saliency'], show_saliency,
                             img_metas, bboxes=gt_bboxes, denorm=False)
            vis_batched_imgs(vis_options['saliency']+'_no_box', show_saliency,
                             img_metas, bboxes=None, denorm=False)

        if self.separable:
            x_saliency = saliency.sum(dim=2)
            y_saliency = saliency.sum(dim=3)
            grid = self.separable_saliency_to_grid(imgs, img_metas, x_saliency,
                                                   y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs, img_metas,
                                                      saliency, device)

        return grid
