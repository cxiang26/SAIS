import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.fcos_head import FCOSHead
INF = 1e8

@HEADS.register_module()
class SAISLHead(FCOSHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(SAISLHead, self).__init__(
            num_classes,
            in_channels,
            regress_ranges=regress_ranges,
            center_sampling=center_sampling,
            center_sample_radius=center_sample_radius,
            norm_on_bbox=norm_on_bbox,
            centerness_on_reg=centerness_on_reg,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)

@HEADS.register_module()
class SAISLSegmHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 loss_segm=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Xavier',
                     distribution='uniform',
                     override=dict(name='segm_conv'))):
        super(SAISLSegmHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_segm = build_loss(loss_segm)
        self._init_layers()
        self.fp16_enabled = False

    def _init_layers(self):
        """Initialize layers of the head."""
        self.segm_conv = nn.Conv2d(
            self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        return self.segm_conv(x)

    @force_fp32(apply_to=('segm_pred', ))
    def loss(self, segm_pred, gt_masks, gt_labels):
        loss_segm = []
        num_imgs, num_classes, mask_h, mask_w = segm_pred.size()
        for idx in range(num_imgs):
            cur_segm_pred = segm_pred[idx]
            cur_gt_masks = gt_masks[idx].float()
            cur_gt_labels = gt_labels[idx]
            segm_targets = self.get_targets(cur_segm_pred, cur_gt_masks,
                                            cur_gt_labels)
            if segm_targets is None:
                loss = self.loss_segm(cur_segm_pred,
                                      torch.zeros_like(cur_segm_pred),
                                      torch.zeros_like(cur_segm_pred))
            else:
                loss = self.loss_segm(
                    cur_segm_pred,
                    segm_targets,
                    avg_factor=num_imgs * mask_h * mask_w)
            loss_segm.append(loss)
        return dict(loss_segm=loss_segm)

    def get_targets(self, segm_pred, gt_masks, gt_labels):
        if gt_masks.size(0) == 0:
            return None
        num_classes, mask_h, mask_w = segm_pred.size()
        with torch.no_grad():
            downsampled_masks = F.interpolate(
                gt_masks.unsqueeze(0), (mask_h, mask_w),
                mode='bilinear',
                align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.gt(0.5).float()
            segm_targets = torch.zeros_like(segm_pred, requires_grad=False)
            for obj_idx in range(downsampled_masks.size(0)):
                segm_targets[gt_labels[obj_idx] - 1] = torch.max(
                    segm_targets[gt_labels[obj_idx] - 1],
                    downsampled_masks[obj_idx])
            return segm_targets

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        raise NotImplementedError(
            'simple_test of YOLACTSegmHead is not implemented '
            'because this head is only evaluated during training')
