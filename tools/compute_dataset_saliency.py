import argparse
import pickle
from os.path import join

import torch
import torch.nn.functional as F
import numpy as np


from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmcv import imwrite
from mmcv.runner import load_checkpoint

import fovea.models  # noqa: F401, add custom modules to MMCV registry
from fovea.utils import mkdir2
from fovea.utils.config import add_config_args, generate_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--weights', type=str, default=None)
    parser = add_config_args(parser)

    opts = parser.parse_args()
    return opts


def main():
    opts = parse_args()

    mkdir2(opts.out_dir)

    cfg = generate_config(opts)

    for split in ['train', 'val', 'test']:
        if split in cfg.data:
            for key in ['av_ann_file', 'att_db', 'ann_file', 'img_prefix']:
                if key in cfg.data[split]:
                    cfg.data[split][key] = join(
                        opts.data_dir, cfg.data[split][key])

    dataset = build_dataset(cfg.data.test)
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    if opts.weights is not None:
        load_checkpoint(model, opts.weights)
    model.eval()

    # concatenate annotations from training set
    bboxes = [
        torch.Tensor(dataset.get_ann_info(i)['bboxes'])
        for i in range(len(dataset))
    ]
    bboxes = torch.cat(bboxes, dim=0)

    # compute dataset-wide saliency
    shape = (1200, 1920, 3)
    saliency = model.grid_net.bbox2sal(
        [bboxes], [{'pad_shape': shape}], jitter=False)

    # visualize dataset-wide saliency
    h, w, _ = shape
    show_saliency = F.interpolate(saliency, size=(h, w), mode='bilinear',
                                  align_corners=True)
    show_saliency = 255*(show_saliency/show_saliency.max())
    show_saliency = show_saliency.expand(show_saliency.size(0), 3, h, w) \
                                 .permute(0, 2, 3, 1)
    show_saliency = np.ascontiguousarray(show_saliency.cpu().detach().numpy())
    imwrite(show_saliency[0], join(opts.out_dir, 'dataset_saliency.jpg'))

    # save dataset-wide saliency
    out_path = join(opts.out_dir, 'dataset_saliency.pkl')
    pickle.dump(saliency, open(out_path, 'wb'))


if __name__ == '__main__':
    main()
