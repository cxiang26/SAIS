# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet.core import bbox2result, mask_target
from mmdet.models.builder import DETECTORS, build_head
from mmdet.models.detectors.single_stage import SingleStageDetector
import torch.nn.functional as F
from mmcv.ops import RoIAlign

@DETECTORS.register_module()
class SAISGL(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 segm_head,
                 mask_head,
                 roi_mask_head,
                 use_ext_feat=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SAISGL, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)
        if segm_head is not None:
            self.segm_head = build_head(segm_head)
        else:
            self.segm_head = segm_head
        self.mask_head = build_head(mask_head)
        self.roi_mask_head = build_head(roi_mask_head)
        self.roialign = RoIAlign(output_size=(14, 14))
        self.use_ext_feat = use_ext_feat
        if use_ext_feat:
            self.ext_conv = torch.nn.Conv2d(256, 256, 3, 1, padding=1)
            self.relu = torch.nn.ReLU(inplace=True)

    def forward_dummy(self, img):
        feat = self.extract_feat(img)
        bbox_outs = self.bbox_head(feat)
        prototypes = self.mask_head.forward_dummy(feat[0])
        return (bbox_outs, prototypes)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # convert Bitmap mask or Polygon Mask to Tensor here
        gt_seg_masks = [
            gt_mask.to_tensor(dtype=torch.uint8, device=img.device)
            for gt_mask in gt_masks
        ]

        x = self.extract_feat(img)

        cls_score, bbox_pred, center_ness, coeff_pred, reg_feat = self.bbox_head(x)
        bbox_head_loss_inputs = (cls_score, bbox_pred, center_ness) + (gt_bboxes, gt_labels,
                                                          img_metas)
        losses, sampling_results = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if self.use_ext_feat:
            in_feat = self.relu(self.ext_conv(x[0]))

        feat_masks = []
        for i, feat in enumerate(reg_feat):
            if i < 3:
                if i == 0:
                    if self.use_ext_feat:
                        feat = feat + in_feat
                    feat_masks.append(feat + in_feat)
                else:
                    feat_up = F.interpolate(feat, scale_factor=(2 ** i), mode='bilinear',
                                            align_corners=False)
                    if self.use_ext_feat:
                        feat_up = feat_up + in_feat
                    feat_masks.append(feat_up)
        feat_masks = torch.cat(feat_masks, dim=1)

        if self.segm_head is not None:
            segm_head_outs = self.segm_head(x[0])
            loss_segm = self.segm_head.loss(segm_head_outs, gt_seg_masks, gt_labels)
            losses.update(loss_segm)

        ## roi loss
        bboxes = [box / 8. for box in gt_bboxes]
        idx = np.concatenate([[i] * len(box) for i, box in enumerate(gt_bboxes)])
        bboxes = torch.cat(bboxes, dim=0)
        idx = torch.from_numpy(idx.astype(np.float32)).to(bboxes.device)
        bboxes_new = torch.cat([idx[:,None], bboxes], dim=1)

        roi_feats = self.roialign(x[0], bboxes_new)
        mask_preds = self.roi_mask_head(roi_feats)

        labels = [l.new_tensor([ii for ii in range(len(l))]) for l in gt_labels]
        mask_targets = mask_target(gt_bboxes, labels, gt_masks, self.train_cfg)
        roi_loss_mask = self.roi_mask_head.loss(mask_preds, mask_targets, torch.cat(gt_labels))
        losses.update(roi_loss_mask)

        mask_pred = self.mask_head(feat_masks, coeff_pred, gt_bboxes, img_metas,
                                   sampling_results)
        loss_mask = self.mask_head.loss(mask_pred, gt_seg_masks, gt_bboxes,
                                        img_metas, sampling_results)
        losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        feats = self.extract_feat(img)
        cls_scores, bbox_preds, centernesses, coefficients, reg_feats = self.bbox_head(feats)
        result_list = self.bbox_head.get_bboxes(cls_scores, bbox_preds, centernesses, coefficients, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label, _ in result_list
        ]

        if self.use_ext_feat:
            in_feat = self.relu(self.ext_conv(feats[0]))

        feat_masks = []
        for i, feat in enumerate(reg_feats):
            if i < 3:
                if i == 0:
                    if self.use_ext_feat:
                        feat = feat + in_feat
                    feat_masks.append(feat)
                else:
                    feat_up = F.interpolate(feat, scale_factor=(2 ** i), mode='bilinear',
                                            align_corners=False)
                    if self.use_ext_feat:
                        feat_up = feat_up + in_feat
                    feat_masks.append(feat_up)
        feat_masks = torch.cat(feat_masks, dim=1)

        segm_results = self.mask_head.simple_test(
            [feat_masks],
            [result_list[0][0]],
            [result_list[0][1]],
            [result_list[0][2]],
            img_metas,
            rescale=rescale)

        return list(zip(bbox_results, segm_results))