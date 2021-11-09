import argparse
import pickle
from os.path import join
from time import perf_counter

from tqdm import trange
import torch
import mmcv
from mmdet.core import encode_mask_results
from mmcv.image import tensor2imgs
from mmdet.datasets import build_dataset
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel, collate, scatter

import fovea.models  # noqa: F401, add custom modules to MMCV registry
from fovea.utils import mkdir2, print_stats, apply_class_map, build_detector
from fovea.utils.config import add_config_args, generate_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--map-classes', action='store_true', default=False)
    parser.add_argument('--use-prevdets', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument(
        '--vis-options',
        nargs='+',
        action=mmcv.DictAction,
        help='custom options for visualization, the key-value pair in xxx=yyy '
        'format will be kwargs for model.visualize() function')
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
                        opts.data_dir,
                        cfg.data[split][key]
                    )

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    print(cfg.pretty_text)
    cfg.dump(join(opts.out_dir, 'config.py'))

    if opts.use_prevdets:
        assert cfg.data.test.type == 'AVHDPrevDetDataset'
    dataset = build_dataset(cfg.data.test)

    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    if opts.weights is not None:
        checkpoint = load_checkpoint(model, opts.weights)
    if opts.weights is not None and 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model.cfg = cfg
    model.eval()

    if opts.gpu_test_pre:
        model = model.cuda()
    else:
        model = MMDataParallel(model, device_ids=[0])

    results = []
    runtimes = []
    if opts.use_prevdets:
        prevdets = None
    for i in trange(len(dataset)):
        if opts.use_prevdets:
            # load from dataset using actual dets from previous iteration
            data = dataset.prepare_test_img_with_prevdets(i, prevdets)
        else:
            data = dataset[i]
        if opts.gpu_test_pre:
            preprocessing_time = \
                perf_counter() - data['img_metas'][0][0]['start_time']
        else:
            preprocessing_time = \
                data['img_metas'][0].data['preprocessing_time']

        start = perf_counter()
        if not opts.gpu_test_pre:
            data = collate([data], samples_per_gpu=1)
            data = scatter(data, ["cuda:0"])[0]

        with torch.no_grad():
            if opts.vis_options:
                result = model(
                    return_loss=False, rescale=True,
                    vis_options=opts.vis_options, **data
                )
            else:
                result = model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        runtimes.append(perf_counter() - start + preprocessing_time)

        batch_size = len(result)
        if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
            img_tensor = data['img'][0]
        else:
            img_tensor = data['img'][0].data
        img_metas = data['img_metas'][0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        # map classes if necessary
        if opts.map_classes:
            num_classes = len(dataset.CLASSES)
            if isinstance(result[0], tuple):
                for i in range(len(result)):
                    result[i] = tuple(
                        apply_class_map(result_type, dataset.class_mapping,
                                        num_classes)
                        for result_type in result[i]
                    )
            else:
                result = [
                    apply_class_map(r, dataset.class_mapping, num_classes)
                    for r in result
                ]

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]

        if opts.use_prevdets:
            prevdets = dataset.result_to_det(result[0])

        results.extend(result)

    # convert to ms for display
    def s2ms(x):
        return 1e3*x
    print_stats(runtimes, 'Total Runtime (ms)', cvt=s2ms)

    out_path = join(opts.out_dir, 'results.pkl')
    pickle.dump(results, open(out_path, 'wb'))

    out_path = join(opts.out_dir, 'time_info.pkl')
    runtime_info = {
        'runtimes': runtimes,
        'runtime_all': runtimes,  # compatibility with rtAP
        'n_total': len(runtimes)
    }
    pickle.dump(runtime_info, open(out_path, 'wb'))

    # for compatibility with rtAP
    out_path = join(opts.out_dir, 'results_ccf.pkl')
    results_ccf = dataset._det2json(results)
    pickle.dump(results_ccf, open(out_path, 'wb'))

    out_path = join(opts.out_dir, 'eval')
    eval_kwargs = cfg.get('evaluation', {}).copy()
    eval_results = dataset.evaluate(
        results, jsonfile_prefix=out_path, **eval_kwargs)
    out_path = join(opts.out_dir, 'eval.pkl')
    pickle.dump(eval_results, open(out_path, 'wb'))


if __name__ == '__main__':
    main()
