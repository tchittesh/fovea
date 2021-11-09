import os.path as osp

import numpy as np
from pycocotools.coco import COCO
from mmdet.datasets.builder import DATASETS
from mmdet.datasets import CocoDataset
from mmdet.datasets.pipelines import Compose


@DATASETS.register_module()
class AVHDDataset(CocoDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 take=None,
                 skip=None):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.take = take
        self.skip = skip

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        db = self.coco.dataset

        self.CLASSES = tuple([c['name'] for c in db['categories']])
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.coco_mapping = db.get('coco_mapping', None)
        self.class_mapping = {
            i: v
            for i, v in enumerate(db.get('coco_mapping', None)) if v < 80
        }

        self.seqs = db['sequences']
        self.seq_dirs = db['seq_dirs']
        self.img_ids = self.coco.getImgIds()

        img_infos = []
        for img in self.coco.imgs.values():
            img_name = img['name']
            sid = img['sid']
            img_path = osp.join(self.seq_dirs[sid], img_name)
            img['filename'] = img_path
            img_infos.append(img)

        if self.skip is not None:
            self.img_ids = self.img_ids[::self.skip]
            img_infos = img_infos[::self.skip]
        if self.take is not None:
            self.img_ids = self.img_ids[:self.take]
            img_infos = img_infos[:self.take]
        return img_infos


@DATASETS.register_module()
class AVHDGTDataset(AVHDDataset):
    """AVHDDataset, but returns images with ground
    truth annotations during testing."""

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_train_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data


@DATASETS.register_module()
class AVHDPrevDetDataset(AVHDDataset):
    """AVHDDataset, but with helpers to facilitate online testing."""

    def result_to_det(self, result):
        if isinstance(result, tuple):
            # contains mask
            result = result[0]

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels).astype(np.int64)
        n_det = len(labels)
        if n_det:
            bboxes = np.vstack(result)
            scores = bboxes[:, -1].astype(np.float32)
            bboxes = bboxes[:, :4].astype(np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

        return dict(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            bboxes_ignore=None,
            masks=n_det*[None],
            seg_map=n_det*[None],
        )

    def prepare_test_img_with_prevdets(self, idx, prevdets):
        """
        Returns image at index 'idx' with:
        - empty annotations if this image is the first frame of its sequence
        - prevdets annotations else
        """
        img_info = self.data_infos[idx]
        pidx = self.get_prev_idx(idx)
        if pidx is None:
            det = dict(
                bboxes=np.zeros((0, 4), dtype=np.float32),
                labels=np.array([], dtype=np.int64),
                scores=np.array([], dtype=np.float32),
                bboxes_ignore=None,
                masks=[],
                seg_map=[],
            )
        else:
            det = prevdets
        results = dict(img_info=img_info, ann_info=det)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_prev_idx(self, idx):
        img = self.data_infos[idx]
        if img['fid'] == 0:
            return None
        else:
            return idx - 1
