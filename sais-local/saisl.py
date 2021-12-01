import torch
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS, build_head
from mmdet.models.detectors import SingleStageDetector

from mmdet.core import bbox2result, mask_target
from mmcv.ops import RoIAlign

import numpy as np

@DETECTORS.register_module()
class SAISL(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 segm_head,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SAISL, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)
        self.segm_head = build_head(segm_head)
        self.mask_head = build_head(mask_head)
        self.roialign = RoIAlign(output_size=(14, 14))

    def forward_dummy(self, img):
        feat = self.extract_feat(img)
        bbox_outs = self.bbox_head(feat)
        prototypes = self.mask_head(feat[0])
        return (bbox_outs, prototypes)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        # convert Bitmap mask or Polygon Mask to Tensor here
        gt_seg_masks = [
            gt_mask.to_tensor(dtype=torch.uint8, device=img.device)
            for gt_mask in gt_masks
        ]

        x = self.extract_feat(img)

        cls_score, bbox_pred, centerness_pred = self.bbox_head(x)
        bbox_head_loss_inputs = (cls_score, bbox_pred, centerness_pred) + (gt_bboxes, gt_labels,
                                                          img_metas)
        losses = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        segm_head_outs = self.segm_head(x[0])
        loss_segm = self.segm_head.loss(segm_head_outs, gt_seg_masks, gt_labels)
        losses.update(loss_segm)

        bboxes = [box / 8. for box in gt_bboxes]
        idx = np.concatenate([[i] * len(box) for i, box in enumerate(gt_bboxes)])
        bboxes = torch.cat(bboxes, dim=0)
        idx = torch.from_numpy(idx.astype(np.float32)).to(bboxes.device)
        bboxes_new = torch.cat([idx[:,None], bboxes], dim=1)

        roi_feats = self.roialign(x[0], bboxes_new)
        mask_preds = self.mask_head(roi_feats)

        labels = [l.new_tensor([ii for ii in range(len(l))]) for l in gt_labels]
        mask_targets = mask_target(gt_bboxes, labels, gt_masks, self.train_cfg)
        loss_mask = self.mask_head.loss(mask_preds, mask_targets, torch.cat(gt_labels))
        losses.update(loss_mask)

        # check NaN and Inf
        # for loss_name in losses.keys():
        #     assert torch.isfinite(torch.stack(losses[loss_name]))\
        #         .all().item(), '{} becomes infinite or NaN!'\
        #         .format(loss_name)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        cls_score, bbox_pred, centerness_pred = self.bbox_head(feat)
        results_list = self.bbox_head.get_bboxes(cls_score, bbox_pred, centerness_pred, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]

        assert len(img_metas) == 1
        bboxes = results_list[0][0][:, :4]
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes  * (scale_factor / 8.)
        idx = bboxes.new_zeros((bboxes.size(0), 1))
        bboxes_new = torch.cat([idx, bboxes], dim=1)
        roi_feats = self.roialign(feat[0], bboxes_new)
        mask_preds = self.mask_head(roi_feats)
        segm_results = self.mask_head.get_seg_masks(mask_preds, results_list[0][0], results_list[0][1],
                                                    self.test_cfg, ori_shape, torch.ones_like(scale_factor), rescale)
        return list(zip(bbox_results, [segm_results]))

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

    def get_crop_feat(self, feats, gt_box):
        pass
