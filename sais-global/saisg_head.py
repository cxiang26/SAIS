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
class SAISGHead(FCOSHead):
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
        super(SAISGHead, self).__init__(
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

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.num_coefficients = 32
        self.conv_coeff = nn.Conv2d(self.feat_channels, self.num_coefficients, 3, padding=1)

    def forward_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        bbox_pred = self.conv_reg(reg_feat)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        coefficients = self.conv_coeff(cls_feat).tanh()
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness, coefficients, reg_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, label_list, bbox_targets_list, gt_inds = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness), \
               dict(label_list=label_list,
                    bbox_targets_list=bbox_targets_list,
                    gt_inds=gt_inds)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, gt_inds = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets, labels_list, bbox_targets_list, gt_inds

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels.new_empty(0)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        gt_inds = min_area_inds[labels < self.num_classes]

        return labels, bbox_targets, gt_inds

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'coefficients'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   coefficients,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        coefficients_list = [coefficients[i].detach() for i in range(num_levels)]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, coefficients_list, mlvl_points,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    coefficients,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_coefficients = []
        for cls_score, bbox_pred, centerness, coefficient, points in zip(
                cls_scores, bbox_preds, centernesses, coefficients, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            coefficient = coefficient.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_coefficients)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[
                                           1] * batch_inds + topk_inds
                    points = points.reshape(-1,
                                            2)[transformed_inds, :].reshape(
                        batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                        batch_size, -1, self.num_classes)
                    centerness = centerness.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]
                    coefficient = coefficient[batch_inds, topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_coefficients.append(coefficient)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
        batch_mlvl_coefficients = torch.cat(mlvl_coefficients, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_mlvl_scores = batch_mlvl_scores * (
                batch_mlvl_centerness.unsqueeze(2))
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label, inds = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness,
                    return_inds=True)
                batch_mlvl_coefficients = batch_mlvl_coefficients.reshape(-1, self.num_coefficients)
                batch_mlvl_coefficients = batch_mlvl_coefficients[:, None, :].expand(batch_mlvl_coefficients.shape[0], self.num_classes, self.num_coefficients)
                batch_mlvl_coefficients = batch_mlvl_coefficients.reshape(-1, self.num_coefficients)
                det_coefficents = batch_mlvl_coefficients[inds, :]
                det_results.append(tuple([det_bbox, det_label, det_coefficents]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results


@HEADS.register_module()
class SAISGSegmHead(BaseModule):
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
        super(SAISGSegmHead, self).__init__(init_cfg)
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

@HEADS.register_module()
class SAISGProtonet(BaseModule):
    """YOLACT mask head used in https://arxiv.org/abs/1904.02689.

    This head outputs the mask prototypes for YOLACT.

    Args:
        in_channels (int): Number of channels in the input feature map.
        proto_channels (tuple[int]): Output channels of protonet convs.
        proto_kernel_sizes (tuple[int]): Kernel sizes of protonet convs.
        include_last_relu (Bool): If keep the last relu of protonet.
        num_protos (int): Number of prototypes.
        num_classes (int): Number of categories excluding the background
            category.
        loss_mask_weight (float): Reweight the mask loss by this factor.
        max_masks_to_train (int): Maximum number of masks to train for
            each image.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 proto_channels=(256, 256, 256, None, 256, 32),
                 proto_kernel_sizes=(3, 3, 3, -2, 3, 1),
                 include_last_relu=True,
                 num_protos=32,
                 loss_mask_weight=1.0,
                 max_masks_to_train=100,
                 init_cfg=dict(
                     type='Xavier',
                     distribution='uniform',
                     override=dict(name='protonet'))):
        super(SAISGProtonet, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.proto_channels = proto_channels
        self.proto_kernel_sizes = proto_kernel_sizes
        self.include_last_relu = include_last_relu
        self.protonet = self._init_layers()

        self.loss_mask_weight = loss_mask_weight
        self.num_protos = num_protos
        self.num_classes = num_classes
        self.max_masks_to_train = max_masks_to_train
        self.fp16_enabled = False

    def _init_layers(self):
        """A helper function to take a config setting and turn it into a
        network."""
        # Possible patterns:
        # ( 256, 3) -> conv
        # ( 256,-2) -> deconv
        # (None,-2) -> bilinear interpolate
        in_channels = self.in_channels
        protonets = ModuleList()

        for num_channels, kernel_size in zip(self.proto_channels,
                                             self.proto_kernel_sizes):
            if kernel_size > 0:
                layer = nn.Conv2d(
                    in_channels,
                    num_channels,
                    kernel_size,
                    padding=kernel_size // 2)
            else:
                if num_channels is None:
                    layer = InterpolateModule(
                        scale_factor=-kernel_size,
                        mode='bilinear',
                        align_corners=False)
                else:
                    layer = nn.ConvTranspose2d(
                        in_channels,
                        num_channels,
                        -kernel_size,
                        padding=kernel_size // 2)
            protonets.append(layer)
            protonets.append(nn.ReLU(inplace=True))
            in_channels = num_channels if num_channels is not None \
                else in_channels
        if not self.include_last_relu:
            protonets = protonets[:-1]
        protonets.append(InterpolateModule(scale_factor=4, mode='bilinear', align_corners=False))
        return nn.Sequential(*protonets)

    def init_weights(self):
        pass

    def forward_dummy(self, x):
        prototypes = self.protonet(x)
        return prototypes

    def forward(self, x, coeff_pred, bboxes, img_meta, sampling_results=None):
        """Forward feature from the upstream network to get prototypes and
        linearly combine the prototypes, using masks coefficients, into
        instance masks. Finally, crop the instance masks with given bboxes.

        Args:
            x (Tensor): Feature from the upstream network, which is
                a 4D-tensor.
            coeff_pred (list[Tensor]): Mask coefficients for each scale
                level with shape (N, num_anchors * num_protos, H, W).
            bboxes (list[Tensor]): Box used for cropping with shape
                (N, num_anchors * 4, H, W). During training, they are
                ground truth boxes. During testing, they are predicted
                boxes.
            img_meta (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            sampling_results (List[:obj:``SamplingResult``]): Sampler results
                for each image.

        Returns:
            list[Tensor]: Predicted instance segmentation masks.
        """
        prototypes = self.protonet(x)
        prototypes = prototypes.permute(0, 2, 3, 1).contiguous()

        num_imgs = x.size(0)

        # The reason for not using self.training is that
        # val workflow will have a dimension mismatch error.
        # Note that this writing method is very tricky.
        # Fix https://github.com/open-mmlab/mmdetection/issues/5978
        is_train_or_val_workflow = (coeff_pred[0].dim() == 4)

        # Train or val workflow
        if is_train_or_val_workflow:
            coeff_pred_list = []
            for coeff_pred_per_level in coeff_pred:
                coeff_pred_per_level = \
                    coeff_pred_per_level.permute(
                        0, 2, 3, 1).reshape(num_imgs, -1, self.num_protos)
                coeff_pred_list.append(coeff_pred_per_level)
            coeff_pred = torch.cat(coeff_pred_list, dim=1)

        mask_pred_list = []
        for idx in range(num_imgs):
            cur_prototypes = prototypes[idx]
            cur_coeff_pred = coeff_pred[idx]
            cur_bboxes = bboxes[idx]
            cur_img_meta = img_meta[idx]

            # Testing state
            if not is_train_or_val_workflow:
                bboxes_for_cropping = cur_bboxes
            else:
                gt_labels = torch.cat(sampling_results['label_list'][idx])
                # gt_bboxes = torch.cat(sampling_results['bbox_targets_list'][idx], dim=0)
                gt_inds = sampling_results['gt_inds'][idx]

                pos_inds = ((gt_labels >= 0) & (gt_labels < self.num_classes)).nonzero().reshape(-1)
                cur_coeff_pred = cur_coeff_pred[pos_inds]
                bboxes_for_cropping = cur_bboxes[gt_inds].clone()

            # Linearly combine the prototypes with the mask coefficients
            mask_pred = cur_prototypes @ cur_coeff_pred.t()
            mask_pred = torch.sigmoid(mask_pred)

            h, w = cur_img_meta['img_shape'][:2]
            bboxes_for_cropping[:, 0] /= w
            bboxes_for_cropping[:, 1] /= h
            bboxes_for_cropping[:, 2] /= w
            bboxes_for_cropping[:, 3] /= h

            mask_pred = self.crop(mask_pred, bboxes_for_cropping)
            mask_pred = mask_pred.permute(2, 0, 1).contiguous()
            mask_pred_list.append(mask_pred)
        return mask_pred_list

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, gt_masks, gt_bboxes, img_meta, sampling_results):
        """Compute loss of the head.

        Args:
            mask_pred (list[Tensor]): Predicted prototypes with shape
                (num_classes, H, W).
            gt_masks (list[Tensor]): Ground truth masks for each image with
                the same shape of the input image.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_meta (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            sampling_results (List[:obj:``SamplingResult``]): Sampler results
                for each image.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_mask = []
        num_imgs = len(mask_pred)
        total_pos = 0
        for idx in range(num_imgs):
            cur_mask_pred = mask_pred[idx]
            cur_gt_masks = gt_masks[idx].float()
            cur_gt_bboxes = gt_bboxes[idx]
            cur_img_meta = img_meta[idx]

            pos_assigned_gt_inds = sampling_results['gt_inds'][idx]
            num_pos = pos_assigned_gt_inds.size(0)
            # Since we're producing (near) full image masks,
            # it'd take too much vram to backprop on every single mask.
            # Thus we select only a subset.
            if num_pos > self.max_masks_to_train:
                perm = torch.randperm(num_pos)
                select = perm[:self.max_masks_to_train]
                cur_mask_pred = cur_mask_pred[select]
                pos_assigned_gt_inds = pos_assigned_gt_inds[select]
                num_pos = self.max_masks_to_train
            total_pos += num_pos

            gt_bboxes_for_reweight = cur_gt_bboxes[pos_assigned_gt_inds]

            mask_targets = self.get_targets(cur_mask_pred, cur_gt_masks,
                                            pos_assigned_gt_inds)
            if num_pos == 0:
                loss = cur_mask_pred.sum() * 0.
            elif mask_targets is None:
                loss = F.binary_cross_entropy(cur_mask_pred,
                                              torch.zeros_like(cur_mask_pred),
                                              torch.zeros_like(cur_mask_pred))
            else:
                cur_mask_pred = torch.clamp(cur_mask_pred, 0, 1)
                loss = F.binary_cross_entropy(
                    cur_mask_pred, mask_targets,
                    reduction='none') * self.loss_mask_weight

                h, w = cur_img_meta['img_shape'][:2]
                gt_bboxes_width = (gt_bboxes_for_reweight[:, 2] -
                                   gt_bboxes_for_reweight[:, 0]) / w
                gt_bboxes_height = (gt_bboxes_for_reweight[:, 3] -
                                    gt_bboxes_for_reweight[:, 1]) / h
                loss = loss.mean(dim=(1,
                                      2)) / gt_bboxes_width / gt_bboxes_height
                loss = torch.sum(loss)
            loss_mask.append(loss)

        if total_pos == 0:
            total_pos += 1  # avoid nan
        loss_mask = [x / total_pos for x in loss_mask]

        return dict(loss_global_mask=loss_mask)

    def get_targets(self, mask_pred, gt_masks, pos_assigned_gt_inds):
        """Compute instance segmentation targets for each image.

        Args:
            mask_pred (Tensor): Predicted prototypes with shape
                (num_classes, H, W).
            gt_masks (Tensor): Ground truth masks for each image with
                the same shape of the input image.
            pos_assigned_gt_inds (Tensor): GT indices of the corresponding
                positive samples.
        Returns:
            Tensor: Instance segmentation targets with shape
                (num_instances, H, W).
        """
        if gt_masks.size(0) == 0:
            return None
        mask_h, mask_w = mask_pred.shape[-2:]
        gt_masks = F.interpolate(
            gt_masks.unsqueeze(0), (mask_h, mask_w),
            mode='bilinear',
            align_corners=False).squeeze(0)
        gt_masks = gt_masks.gt(0.5).float()
        mask_targets = gt_masks[pos_assigned_gt_inds]
        return mask_targets

    def get_seg_masks(self, mask_pred, label_pred, img_meta, rescale):
        """Resize, binarize, and format the instance mask predictions.

        Args:
            mask_pred (Tensor): shape (N, H, W).
            label_pred (Tensor): shape (N, ).
            img_meta (dict): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If rescale is False, then returned masks will
                fit the scale of imgs[0].
        Returns:
            list[ndarray]: Mask predictions grouped by their predicted classes.
        """
        ori_shape = img_meta['ori_shape']
        scale_factor = img_meta['scale_factor']
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor[1]).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor[0]).astype(np.int32)

        cls_segms = [[] for _ in range(self.num_classes)]
        if mask_pred.size(0) == 0:
            return cls_segms

        mask_pred = F.interpolate(
            mask_pred.unsqueeze(0), (img_h, img_w),
            mode='bilinear',
            align_corners=False).squeeze(0) > 0.5
        mask_pred = mask_pred.cpu().numpy().astype(np.uint8)

        for m, l in zip(mask_pred, label_pred):
            cls_segms[l].append(m)
        return cls_segms

    def crop(self, masks, boxes, padding=1):
        """Crop predicted masks by zeroing out everything not in the predicted
        bbox.

        Args:
            masks (Tensor): shape [H, W, N].
            boxes (Tensor): bbox coords in relative point form with
                shape [N, 4].

        Return:
            Tensor: The cropped masks.
        """
        h, w, n = masks.size()
        x1, x2 = self.sanitize_coordinates(
            boxes[:, 0], boxes[:, 2], w, padding, cast=False)
        y1, y2 = self.sanitize_coordinates(
            boxes[:, 1], boxes[:, 3], h, padding, cast=False)

        rows = torch.arange(
            w, device=masks.device, dtype=x1.dtype).view(1, -1,
                                                         1).expand(h, w, n)
        cols = torch.arange(
            h, device=masks.device, dtype=x1.dtype).view(-1, 1,
                                                         1).expand(h, w, n)

        masks_left = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up = cols >= y1.view(1, 1, -1)
        masks_down = cols < y2.view(1, 1, -1)

        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask.float()

    def sanitize_coordinates(self, x1, x2, img_size, padding=0, cast=True):
        """Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
        and x2 <= image_size. Also converts from relative to absolute
        coordinates and casts the results to long tensors.

        Warning: this does things in-place behind the scenes so
        copy if necessary.

        Args:
            _x1 (Tensor): shape (N, ).
            _x2 (Tensor): shape (N, ).
            img_size (int): Size of the input image.
            padding (int): x1 >= padding, x2 <= image_size-padding.
            cast (bool): If cast is false, the result won't be cast to longs.

        Returns:
            tuple:
                x1 (Tensor): Sanitized _x1.
                x2 (Tensor): Sanitized _x2.
        """
        x1 = x1 * img_size
        x2 = x2 * img_size
        if cast:
            x1 = x1.long()
            x2 = x2.long()
        x1 = torch.min(x1, x2)
        x2 = torch.max(x1, x2)
        x1 = torch.clamp(x1 - padding, min=0)
        x2 = torch.clamp(x2 + padding, max=img_size)
        return x1, x2

    def simple_test(self,
                    feats,
                    det_bboxes,
                    det_labels,
                    det_coeffs,
                    img_metas,
                    rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
               upstream network, each is a 4D-tensor.
            det_bboxes (list[Tensor]): BBox results of each image. each
               element is (n, 5) tensor, where 5 represent
               (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
            det_labels (list[Tensor]): BBox results of each image. each
               element is (n, ) tensor, each element represents the class
               label of the corresponding box.
            det_coeffs (list[Tensor]): BBox coefficient of each image. each
               element is (n, m) tensor, m is vector length.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.
        """
        num_imgs = len(img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_preds = self.forward(feats[0], det_coeffs, _bboxes, img_metas)
            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append([[] for _ in range(self.num_classes)])
                else:
                    segm_result = self.get_seg_masks(mask_preds[i],
                                                     det_labels[i],
                                                     img_metas[i], rescale)
                    segm_results.append(segm_result)
        return segm_results


class InterpolateModule(BaseModule):
    """This is a module version of F.interpolate.

    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args, init_cfg=None, **kwargs):
        super().__init__(init_cfg)

        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        """Forward features from the upstream network."""
        return F.interpolate(x, *self.args, **self.kwargs)