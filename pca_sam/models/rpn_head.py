"""Numerically safe proposal postprocessing for Windows mixed-precision training."""
from mmdet.models.dense_heads.rpn_head import RPNHead
from mmdet.registry import MODELS


@MODELS.register_module()
class PCASAMRPNHead(RPNHead):
    def _bbox_post_process(self, results, cfg, rescale=False, with_nms=True, img_meta=None):
        # MMCV's Windows NMS kernel returns float32 scores, including for AMP inputs.
        results.bboxes = results.bboxes.float()
        results.scores = results.scores.float()
        return super()._bbox_post_process(results, cfg, rescale=rescale,
                                         with_nms=with_nms, img_meta=img_meta)
