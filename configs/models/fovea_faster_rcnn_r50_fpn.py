_base_ = './faster_rcnn_r50_fpn.py'
model = dict(
    type='FoveaFasterRCNN',
    rpn_head=dict(
        type='FoveaRPNHead',
    ),
    roi_head=dict(
        type='FoveaStandardRoIHead',
        bbox_head=dict(
            type='FoveaShared2FCBBoxHead',
        )
    ))
