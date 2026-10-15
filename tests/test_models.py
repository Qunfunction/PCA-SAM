"""Focused checks for the released PCA-SAM module contracts."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import register_all_modules

import pca_sam
from pca_sam.models import CSPCNeck

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        register_all_modules(init_default_scope=True)
        cls.cfg = Config.fromfile(str(ROOT / 'configs/pca_sam_vit_b_lphhg.py'))

    def small_head(self):
        cfg = copy.deepcopy(self.cfg.model.roi_head)
        for box in cfg.bbox_head:
            box.in_channels = 8
            box.conv_out_channels = 8
            box.fc_out_channels = 32
        cfg.bbox_roi_extractor.out_channels = 8
        cfg.mask_roi_extractor.out_channels = 8
        cfg.mask_head.in_channels = 16
        cfg.mask_head.conv_out_channels = 16
        cfg.sam_mask_head.in_channels = 8
        cfg.sam_mask_head.conv_out_channels = 8
        cfg.mask_decoder = None
        cfg.prompt_encoder = None
        cfg.train_cfg = self.cfg.model.train_cfg.rcnn
        cfg.test_cfg = copy.deepcopy(self.cfg.model.test_cfg.rcnn)
        cfg.test_cfg.score_thr = 0.0
        cfg.test_cfg.max_per_img = 10
        return MODELS.build(cfg)

    def test_config_registration(self):
        self.assertEqual(self.cfg.model.type, 'PCASAM')
        self.assertEqual(self.cfg.model.neck.type, 'CSPCNeck')
        self.assertEqual(self.cfg.model.roi_head.type, 'CBRDHead')
        for name in ['PCASAM', 'CSPCNeck', 'CBRDHead', 'PCASAMImageEncoder',
                     'SAMTeacherMaskDecoder', 'SAMTeacherPromptEncoder']:
            self.assertIsNotNone(MODELS.get(name))
        self.assertEqual(self.cfg.model.roi_head.stage_loss_weights, [1, .5, .25])

    def test_cspc_shapes_and_gradients(self):
        neck = CSPCNeck(16, 8, target_sizes=((32, 32), (16, 16), (8, 8), (4, 4)))
        inputs = [torch.randn(1, 8, 8, 8, requires_grad=True) for _ in range(4)]
        outputs = neck(inputs)
        self.assertEqual([tuple(t.shape) for t in outputs],
                         [(1, 256, n, n) for n in (32, 16, 8, 4, 2)])
        sum(t.square().mean() for t in outputs).backward()
        self.assertTrue(all(x.grad is not None for x in inputs))
        self.assertTrue(torch.isfinite(neck.cross_attention.in_proj_weight.grad).all())

    def test_frozen_sam_components_without_initialization_files(self):
        encoder = MODELS.build(dict(type='PCASAMImageEncoder', img_size=64,
                                   embed_dim=32, depth=4, num_heads=4,
                                   global_attn_indexes=[0, 1, 2, 3], out_chans=256))
        prompt = MODELS.build(dict(type='SAMTeacherPromptEncoder', embed_dim=256,
                                  image_embedding_size=(4, 4), input_image_size=(64, 64),
                                  mask_in_chans=16))
        decoder = MODELS.build(dict(type='SAMTeacherMaskDecoder', transformer_dim=256))
        for module in [encoder, prompt, decoder]:
            self.assertFalse(any(p.requires_grad for p in module.parameters()))
        features = encoder(torch.randn(1, 3, 64, 64))
        self.assertEqual(len(features), 5)
        self.assertEqual(tuple(features[-1].shape), (1, 256, 4, 4))

    def test_student_and_boundary_masks(self):
        head = self.small_head()
        x = torch.randn(2, 8, 14, 14, requires_grad=True)
        auxiliary, intermediate = head.sam_mask_head[0](x)
        final = head.mask_head[0](x, intermediate)
        self.assertEqual(tuple(auxiliary.shape), (2, 1, 28, 28))
        self.assertEqual(tuple(intermediate.shape), (2, 8, 14, 14))
        self.assertEqual(tuple(final.shape), (2, 2, 28, 28))
        (auxiliary.square().mean() + final.square().mean()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_contextual_box_prediction(self):
        head = self.small_head()
        obj = torch.randn(2, 8, 7, 7)
        context = torch.randn(2, 8, 7, 7)
        cls, box, context_cls, context_box = head.bbox_head[0](obj, context)
        self.assertEqual(tuple(cls.shape), (2, 3))
        self.assertEqual(tuple(box.shape), (2, 4))
        self.assertEqual(tuple(context_cls.shape), (2, 3))
        self.assertEqual(tuple(context_box.shape), (2, 4))

    def test_teacher_free_inference(self):
        head = self.small_head().eval()
        head.gen_mask = Mock(side_effect=AssertionError('Teacher called during inference'))
        features = tuple(torch.randn(1, 8, n, n) for n in (16, 8, 4, 2))
        proposals = [InstanceData(bboxes=torch.tensor([[8., 8., 32., 32.],
                                                       [24., 24., 48., 48.]]),
                                  scores=torch.tensor([.9, .8]))]
        sample = DetDataSample(metainfo=dict(img_shape=(64, 64), ori_shape=(64, 64),
                                             scale_factor=(1., 1.)))
        with torch.no_grad():
            result = head.predict(features, proposals, [sample],
                                  torch.randn(1, 256, 4, 4), rescale=True)
        head.gen_mask.assert_not_called()
        self.assertEqual(tuple(result[0].masks.shape[1:]), (64, 64))
        self.assertEqual(len(result[0].bboxes), len(result[0].masks))

    def test_real_teacher_target_alignment(self):
        head = self.small_head()
        head.prompt_encoder = MODELS.build(dict(
            type='SAMTeacherPromptEncoder', embed_dim=256,
            image_embedding_size=(64, 64), input_image_size=(1024, 1024),
            mask_in_chans=16))
        head.mask_decoder = MODELS.build(dict(type='SAMTeacherMaskDecoder', transformer_dim=256))
        rois = torch.tensor([[0., 100., 100., 300., 300.],
                             [0., 350., 350., 550., 550.]])
        targets = head.gen_mask(rois, torch.randn(1, 256, 64, 64))
        self.assertEqual(tuple(targets.shape), (2, 1, 28, 28))
        self.assertTrue(torch.isfinite(targets).all())
        self.assertFalse(targets.requires_grad)

    def test_cascade_training_losses(self):
        head = self.small_head().train()
        # Test loss plumbing separately from the frozen teacher's expensive forward.
        head.gen_mask = lambda rois, image_embeddings: torch.ones(
            len(rois), 1, 28, 28, device=rois.device)
        features = tuple(torch.randn(1, 8, n, n, requires_grad=True)
                         for n in (16, 8, 4, 2))
        proposals = [InstanceData(bboxes=torch.tensor([[8., 8., 32., 32.],
                                                       [24., 24., 48., 48.]]),
                                  scores=torch.tensor([.9, .8]))]
        masks = np.zeros((1, 64, 64), dtype=np.uint8)
        masks[0, 8:32, 8:32] = 1
        sample = DetDataSample(metainfo=dict(img_shape=(64, 64), ori_shape=(64, 64),
                                             scale_factor=(1., 1.)))
        sample.gt_instances = InstanceData(bboxes=torch.tensor([[8., 8., 32., 32.]]),
                                            labels=torch.tensor([0]),
                                            masks=BitmapMasks(masks, 64, 64))
        sample.ignored_instances = InstanceData(bboxes=torch.empty(0, 4))
        losses = head.loss(features, proposals, [sample], torch.randn(1, 256, 4, 4))
        for stage in range(3):
            for suffix in ['loss_cls', 'loss_bbox', 'loss_cls_context',
                           'loss_bbox_context', 'loss_mask', 'loss_sam_mask']:
                self.assertIn(f's{stage}.{suffix}', losses)
                self.assertTrue(torch.isfinite(losses[f's{stage}.{suffix}']).all())
        total = sum(value for name, value in losses.items() if 'loss' in name)
        total.backward()
        self.assertIsNotNone(features[0].grad)


if __name__ == '__main__':
    unittest.main()
