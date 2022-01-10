_base_ = './saisgls++_r50_caffe_fpn_2x_704.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet101_caffe')))