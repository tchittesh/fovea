import argparse
import copy
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
from mmcv import DictAction
from mmcv.runner import init_dist
from mmdet import __version__ as mmdet_version
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.utils import collect_env, get_root_logger

import fovea.models  # noqa: F401, add custom modules to MMCV registry
from fovea.utils import get_mmdet_hash, build_detector
from fovea.utils.config import add_config_args, generate_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--data-dir')
    parser.add_argument('--ssd-dir')
    parser.add_argument('--load-from', type=str, default=None)
    parser.add_argument(
        '--work-dir', required=True,
        help='the dir to savelogs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--vis-options',
        nargs='+',
        action=DictAction,
        help='custom options for visualization, the key-value pair in xxx=yyy '
        'format will be kwargs for model.visualize() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser = add_config_args(parser)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    if not args.ssd_dir:
        args.ssd_dir = args.data_dir
    torch.autograd.set_detect_anomaly(True)
    # load config
    cfg = generate_config(args)
    for split in ['train', 'val', 'test']:
        if split in cfg.data:
            for key in ['root', 'img_prefix']:
                if key in cfg.data[split]:
                    cfg.data[split][key] = osp.join(args.ssd_dir, cfg.data[split][key])  # noqa: E501
            for key in ['av_ann_file', 'att_db', 'ann_file']:
                if key in cfg.data[split]:
                    cfg.data[split][key] = osp.join(args.data_dir, cfg.data[split][key])  # noqa: E501

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, 'config.py'))
    # clear tf_logs directory
    if osp.exists(osp.join(cfg.work_dir, 'tf_logs')):
        print(osp.join(cfg.work_dir, 'tf_logs'))
        shutil.rmtree(osp.join(cfg.work_dir, 'tf_logs'))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info_dict['MMDetection'] = mmdet_version + '+' + get_mmdet_hash()[:7]
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    # meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    if args.vis_options:
        model.vis_options = args.vis_options

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset, take=args.take))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version + get_mmdet_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
