import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from torch import Tensor
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead
from mmdet.models.losses import accuracy
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from typing import List, Optional, Tuple, Union


@MODELS.register_module()
class CBRDContextBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels: int = 1024,
                 *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_convs=0,
            num_reg_fcs=2,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.num_shared_context_convs = 4
        self.num_shared_context_fcs = 1
        self.num_cls_context_convs = 0
        self.num_cls_context_fcs = 0
        self.num_reg_context_convs = 0
        self.num_reg_context_fcs = 0
        self.init_context_bbox_head()
        self.init_bbox_head()

    def init_bbox_head(self):
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=self.cls_last_dim*2, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else \
                box_dim * self.num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=self.reg_last_dim*2, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)

    def init_context_bbox_head(self, scale=1):
        self.shared_context_convs, self.shared_context_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_context_convs, self.num_shared_context_fcs, self.in_channels,
                True, scale=scale)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_context_convs, self.cls_context_fcs, self.cls_context_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_context_convs, self.num_cls_context_fcs, self.shared_out_channels, scale=scale)

        # add reg specific branch
        self.reg_context_convs, self.reg_context_fcs, self.reg_context_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_context_convs, self.num_reg_context_fcs, self.shared_out_channels, scale=scale)

        if self.num_shared_context_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_context_fcs == 0:
                self.cls_context_last_dim *= (self.roi_feat_area*scale*scale)
            if self.num_reg_context_fcs == 0:
                self.reg_context_last_dim *= (self.roi_feat_area*scale*scale)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            cls_context_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_context_predictor_cfg_.update(
                in_features=self.cls_context_last_dim, out_features=cls_channels)
            self.fc_context_cls = MODELS.build(cls_context_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else \
                box_dim * self.num_classes
            reg_context_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_context_predictor_cfg_, (dict, ConfigDict)):
                reg_context_predictor_cfg_.update(
                    in_features=self.reg_context_last_dim, out_features=out_dim_reg)
            self.fc_context_reg = MODELS.build(reg_context_predictor_cfg_)

    def _add_conv_fc_branch(self,
                            num_branch_convs: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False,
                            scale: int = 1) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= (self.roi_feat_area * scale * scale)
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        cls_context_score: Tensor,
                        bbox_context_pred: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        losses = self.loss(
            cls_score,
            bbox_pred,
            cls_context_score,
            bbox_context_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             cls_context_score: Tensor,
             bbox_context_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if cls_context_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_context_score.numel() > 0:
                loss_cls_context_ = self.loss_cls(
                    cls_context_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_context_, dict):
                    losses.update(loss_cls_context_)
                else:
                    losses['loss_cls_context'] = loss_cls_context_
                if self.custom_activation:
                    acc_context_ = self.loss_cls.get_accuracy(cls_context_score, labels)
                    losses.update(acc_context_)
                else:
                    losses['acc_context'] = accuracy(cls_context_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        if bbox_context_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_context_pred = self.bbox_coder.decode(rois[:, 1:], bbox_context_pred)
                    bbox_context_pred = get_box_tensor(bbox_context_pred)
                if self.reg_class_agnostic:
                    pos_bbox_context_pred = bbox_context_pred.view(
                        bbox_context_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_context_pred = bbox_context_pred.view(
                        bbox_context_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox_context'] = self.loss_bbox(
                    pos_bbox_context_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox_context'] = bbox_context_pred[pos_inds].sum()

        return losses

    def _forward_context(self, context: Tensor) -> Tuple[Tensor]:
        if self.num_shared_context_convs > 0:
            for conv in self.shared_context_convs:
                context = conv(context)
        if self.num_shared_context_fcs > 0:
            if self.with_avg_pool:
                context = self.avg_pool(context)
            context = context.flatten(1)
            for fc in self.shared_context_fcs:
                context = self.relu(fc(context))

        context_cls = context
        context_reg = context

        for conv in self.cls_context_convs:
            context_cls = conv(context_cls)
        if context_cls.dim() > 2:
            if self.with_avg_pool:
                context_cls = self.avg_pool(context_cls)
            context_cls = context_cls.flatten(1)
        for fc in self.cls_context_fcs:
            context_cls = self.relu(fc(context_cls))

        for conv in self.reg_context_convs:
            context_reg = conv(context_reg)
        if context_reg.dim() > 2:
            if self.with_avg_pool:
                context_reg = self.avg_pool(context_reg)
            context_reg = context_reg.flatten(1)
        for fc in self.reg_context_fcs:
            context_reg = self.relu(fc(context_reg))

        cls_context_score = self.fc_context_cls(context_cls) if self.with_cls else None
        bbox_context_pred = self.fc_context_reg(context_reg) if self.with_reg else None
        return context_cls, context_reg, cls_context_score, bbox_context_pred

    def _forward_shared(self, x: Tensor) -> Tensor:
        """Forward function for shared part.

        Args:
            x (Tensor): Input feature.

        Returns:
            Tensor: Shared feature.
        """
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        return x

    def _forward_cls_reg(self, x: Tensor, context: Tensor) -> Tuple[Tensor]:
        """Forward function for classification and regression parts.

        Args:
            x (Tensor): Input feature.

        Returns:
            tuple[Tensor]:

                - cls_score (Tensor): classification prediction.
                - bbox_pred (Tensor): bbox prediction.
        """
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        context_cls, context_reg, cls_context_score, bbox_context_pred = self._forward_context(context)
        x_cls = torch.cat([x_cls, context_cls], dim=1)
        x_reg = torch.cat([x_reg, context_reg], dim=1)
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred, cls_context_score, bbox_context_pred

    def forward(self, x: Tuple[Tensor],
                context: Optional[Tensor] = None,
                return_shared_feat: bool = False) -> tuple:
        x_shared = self._forward_shared(x)
        out = self._forward_cls_reg(x_shared, context)

        if return_shared_feat:
            out += (x_shared,)

        return out
