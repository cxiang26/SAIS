_base_ =  './saisgl_r50_caffe_fpn_1x.py'

model = dict(
    type='SAISGLS',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    bbox_head=dict(
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    ),
    roi_mask_head=dict(
        type='FCNMaskHeadS',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=80,
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=0.2)),
    mask_head=dict(
        type='SAISGProtonet',
        in_channels=256,
        proto_channels=(256, 32),
        proto_kernel_sizes=(3, 1),
        num_protos=32,
        num_classes=80,
        max_masks_to_train=100,
        up_scale=4,
        loss_mask_weight=1.),
    # training and testing settings
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.6)))

lr_config = dict(
    warmup='linear',
    step=[20, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)