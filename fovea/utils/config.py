import mmcv


def add_config_args(parser):
    cfg_parser = parser.add_argument_group(
        'config options',
        description='command-line modifications to the config file')
    cfg_parser.add_argument('--dataset-config', type=str, required=True)
    cfg_parser.add_argument('--model-config', type=str, required=True)
    cfg_parser.add_argument('--schedule-config', type=str, required=True)
    cfg_parser.add_argument('--runtime-config', type=str, required=True)

    cfg_parser.add_argument(
        '--preprocess-scale', type=str, default=None,
        help="scale of preprocessed input images")
    cfg_parser.add_argument(
        '--take', type=int, default=None,
        help="take the first N elements of the dataset")
    cfg_parser.add_argument(
        '--skip', type=int, default=None,
        help="take 1 out of every N elements of the dataset")

    cfg_parser.add_argument(
        '--num-classes', type=int, default=80,
        help="number of classes predicted by the model")
    cfg_parser.add_argument(
        '--reg-decoded-bbox', action='store_true', default=False,
        help="decode bboxes and use regular IoU loss")
    cfg_parser.add_argument(
        '--use-fovea', action='store_true', default=False,
        help="wrap the model with the FOVEA pipeline")
    cfg_parser.add_argument(
        '--grid-net-cfg', type=str, default=None,
        help="config of grid generator to use with FOVEA")
    cfg_parser.add_argument(
        '--saliency-file', type=str, default=None,
        help="specify the saliency file to use if using a \
        Composite KDE grid generator.")
    cfg_parser.add_argument(
        '--gpu-test-pre', action='store_true', default=False,
        help="use GPU preprocessing for testing")
    cfg_parser.add_argument(
        '--load-task-weights', type=str, default=None,
        help="weights file for FOVEA's task detector")
    cfg_parser.add_argument(
        '--gridnet-lr-mult', type=float, default=None,
        help="learning rate (and decay) multiplier for \
        gridnet component of FOVEA")
    cfg_parser.add_argument(
        '--gridnet-wd-mult', type=float, default=None,
        help="weight decay multiplier for gridnet \
        component of FOVEA (overrides gridnet-lr-mult)")
    cfg_parser.add_argument(
        '--lr', type=float, default=None,
        help="change the learning rate")
    cfg_parser.add_argument(
        '--weight-decay', type=float, default=None,
        help="change the weight decay")

    return parser


def generate_config(args):
    cfg = generate_model_config(args)
    cfg.update(generate_dataset_config(args))
    cfg.update(generate_runtime_config(args))
    cfg.update(generate_schedule_config(args))

    return cfg


def generate_schedule_config(args):
    """Generates the schedule config given the parsed command line args."""
    cfg = mmcv.Config.fromfile(args.schedule_config)

    if args.lr is not None:
        cfg.optimizer['lr'] = args.lr
    if args.weight_decay is not None:
        cfg.optimizer['weight_decay'] = args.weight_decay

    custom_keys = {}
    if args.gridnet_lr_mult is not None:
        if args.gridnet_wd_mult is not None:
            decay_mult = args.gridnet_wd_mult
        else:
            decay_mult = args.gridnet_lr_mult
        custom_keys['grid_net'] = dict(lr_mult=args.gridnet_lr_mult,
                                       decay_mult=decay_mult)
    if len(custom_keys) > 0:
        cfg.optimizer['paramwise_cfg'] = dict(custom_keys=custom_keys)

    return cfg


def generate_runtime_config(args):
    """Generates the runtime config given the parsed command line args."""
    cfg = mmcv.Config.fromfile(args.runtime_config)
    return cfg


def generate_model_config(args):
    """Generates the model config given the parsed command line args."""
    cfg = mmcv.Config.fromfile(args.model_config)

    assert cfg.model.type in ('FasterRCNN', 'FoveaFasterRCNN')

    cfg.model.roi_head.bbox_head.num_classes = args.num_classes

    if args.reg_decoded_bbox:
        cfg.model.roi_head.bbox_head['reg_decoded_bbox'] = True
        cfg.model.roi_head.bbox_head['loss_bbox'] = dict(
            type='GIoULoss', loss_weight=10.0)
        cfg.model.rpn_head['reg_decoded_bbox'] = True
        cfg.model.rpn_head['loss_bbox'] = dict(
            type='GIoULoss', loss_weight=1.0)

    if args.use_fovea:
        if args.load_task_weights is not None:
            cfg.model['load_from'] = args.load_task_weights
        grid_net = mmcv.Config.fromfile(args.grid_net_cfg).grid_generator
        if args.saliency_file:
            assert grid_net.type in ('CompositeKDEGrid', 'FixedKDEGrid')
            if grid_net.type == 'CompositeKDEGrid':
                grid_net.fixed_grid.saliency_file = args.saliency_file
            if grid_net.type == 'FixedKDEGrid':
                grid_net.saliency_file = args.saliency_file
        cfg.model = dict(type='FOVEAWarp', task_detector=cfg.model,
                         grid_net=grid_net)

    if args.use_fovea:
        cfg.train_cfg.rcnn.sampler.add_gt_as_proposals = False

    return cfg


def generate_dataset_config(args):
    """Generates the dataset config given the parsed command line args.
    Note: Only Argoverse-HD is supported as of now."""
    cfg = mmcv.Config.fromfile(args.dataset_config)

    if args.take is not None:
        cfg.data.train.take = args.take
        cfg.data.val.take = args.take
        cfg.data.test.take = args.take
    if args.skip is not None:
        cfg.data.train.skip = args.skip
        cfg.data.val.skip = args.skip
        cfg.data.test.skip = args.skip

    if args.gpu_test_pre:
        if args.use_fovea:
            size_divisor = None
        else:
            size_divisor = cfg.data.test.pipeline[-1] \
                              .transforms[0].transforms[3]['size_divisor']
        if args.preprocess_scale is not None:
            scale_factor = eval(args.preprocess_scale)
        else:
            raise NotImplementedError
        cfg.data.test.pipeline[-1].transforms = [dict(
            type='ImageTestTransformGPU',
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=size_divisor,
            scale_factor=scale_factor,
        )]
        return cfg

    # Changes to data pipelines

    if args.preprocess_scale is not None:
        scale = eval(args.preprocess_scale)
        new_shape = tuple(int(scale * dim) for dim in cfg.original_image_shape)
        t = dict(type='Resize', img_scale=new_shape, keep_ratio=True)
        idx = 5 if cfg.get('model_type', None) == 'YOLOV3' else 2
        cfg.data.train.pipeline[idx] = t
        cfg.data.val.pipeline[-1].transforms[0].transforms[0] = t
        cfg.data.val.pipeline[-1].img_scale = new_shape
        cfg.data.test.pipeline[-1].transforms[0].transforms[0] = t
        cfg.data.test.pipeline[-1].img_scale = new_shape

    if args.use_fovea:
        # get rid of padding
        cfg.data.train.pipeline.pop(5)
        cfg.data.val.pipeline[-1].transforms[0].transforms.pop(3)
        cfg.data.test.pipeline[-1].transforms[0].transforms.pop(3)

    if args.preprocess_scale is not None and scale == 1:
        # replace Resize transform with dummy that only adds img metadata
        idx = 5 if cfg.get('model_type', None) == 'YOLOV3' else 2
        cfg.data.train.pipeline[idx] = dict(type='DummyResize')
        cfg.data.val.pipeline[-1].transforms[0].transforms[0] = \
            dict(type='DummyResize')
        cfg.data.test.pipeline[-1].transforms[0].transforms[0] = \
            dict(type='DummyResize')

    return cfg
