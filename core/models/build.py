from functools import partial

import torch
import torch.nn as nn

from loguru import logger

from .copy_detector import (
    CopyDetector,
    DINOCopyDetector,
    ClassificationCopyDetector,
    GazeClassificationCopyDetector,
    LocalVerificationClassificationCopyDetector,
    LocalVerificationCopyDescriptor,
    CopyDescriptor,
    LocalVerificationDINO,
)
from .vision_transformer import VisionTransformer
from .fusion import Fusion
from .heads import (
    ClassificationHead,
    GazeClassificationHead,
    FusionHead,
    ProjectionHead,
    LinearHead,
)
from .postprocessors import (
    HistogramMatching,
    Compose,
)
from .resnet import ResNet, Bottleneck
from .decoder import Linear, MaskTransformer
from core.layers import LayerScaleInitBlock, Block, ConvStem, Attention
from core.utils import utils
from timm.models.vision_transformer import PatchEmbed, Mlp

from .. import distributed, dinos_pretrain_path
from ..utils.serialization import Checkpointer


def build_encoder(cfg, **kwargs):
    encoder = VisionTransformer(
        img_size=cfg.input.img_size,
        patch_size=cfg.model.patch_size,
        in_chans=3,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.encoder_depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        qkv_bias=cfg.model.qkv_bias,
        qk_scale=cfg.model.qk_scale,
        drop_rate=cfg.model.drop_rate,
        attn_drop_rate=cfg.model.attn_drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-4),
        block_layers=LayerScaleInitBlock if cfg.model.use_layer_scale else Block,
        patch_layer=ConvStem if cfg.model.use_convstem else PatchEmbed,
        act_layer=nn.GELU,
        attention_block=Attention,
        mlp_block=Mlp,
        init_scale=cfg.model.init_scale,
        **kwargs
    )
    if cfg.model.freeze_patch:
        encoder.patch_embed.requires_grad_(False)
    return encoder


def build_cnn(cfg, **kwargs):
    cnn = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return cnn


def build_fusion(cfg, **kwargs):
    fusion = Fusion(
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.fusion_depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        qkv_bias=cfg.model.qkv_bias,
        qk_scale=cfg.model.qk_scale,
        drop_rate=cfg.model.drop_rate,
        attn_drop_rate=cfg.model.attn_drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-4),
        block_layers=LayerScaleInitBlock if cfg.model.use_layer_scale else Block,
        act_layer=nn.GELU,
        attention_block=Attention,
        mlp_block=Mlp,
        init_scale=cfg.model.init_scale,
        cls_token=True,
        **kwargs
    )

    return fusion


def build_decoder(cfg, **kwargs):
    decoder = MaskTransformer(
        num_cls=2,
        img_size=cfg.input.img_size,
        patch_size=cfg.model.patch_size,
        encoder_dim=cfg.model.embed_dim,
        block_dim=cfg.model.embed_dim,
        num_layers=cfg.model.decoder_depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        qkv_bias=cfg.model.qkv_bias,
        qk_scale=cfg.model.qk_scale,
        drop=cfg.model.drop_rate,
        attn_drop=cfg.model.attn_drop_rate,
        drop_path=cfg.model.drop_path_rate,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-4),
        attention_block=Attention,
        mlp_block=Mlp,
        scale=cfg.model.init_scale,
        **kwargs,
    )

    return decoder


def build_heads(cfg, **kwargs):
    __head_factory__ = {
        "fusion_head": partial(
            FusionHead,
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            qkv_bias=cfg.model.qkv_bias,
            qk_scale=cfg.model.qk_scale,
            drop_rate=cfg.model.drop_rate,
            attn_drop_rate=cfg.model.attn_drop_rate,
            drop_path_rate=cfg.model.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-4),
            block_layers=LayerScaleInitBlock if cfg.model.use_layer_scale else Block,
            act_layer=nn.GELU,
            attention_block=Attention,
            mlp_block=Mlp,
            init_scale=cfg.model.init_scale,
        ),
        "cls_head": partial(
            ClassificationHead,
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            qkv_bias=cfg.model.qkv_bias,
            qk_scale=cfg.model.qk_scale,
            drop_rate=cfg.model.drop_rate,
            attn_drop_rate=cfg.model.attn_drop_rate,
            drop_path_rate=cfg.model.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-4),
            block_layers=LayerScaleInitBlock if cfg.model.use_layer_scale else Block,
            act_layer=nn.GELU,
            attention_block=Attention,
            mlp_block=Mlp,
            init_scale=cfg.model.init_scale,
        ),
        "gaze_cls_head": partial(
            GazeClassificationHead,
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            qkv_bias=cfg.model.qkv_bias,
            qk_scale=cfg.model.qk_scale,
            drop_rate=cfg.model.drop_rate,
            attn_drop_rate=cfg.model.attn_drop_rate,
            drop_path_rate=cfg.model.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-4),
            block_layers=LayerScaleInitBlock if cfg.model.use_layer_scale else Block,
            act_layer=nn.GELU,
            attention_block=Attention,
            mlp_block=Mlp,
            init_scale=cfg.model.init_scale,
        ),
        "proj_head": partial(
            ProjectionHead,
            encoder_embed_dim=cfg.model.embed_dim
        ),
        "linear_head": partial(
            LinearHead,
            embed_dim=cfg.model.embed_dim
        ),
    }

    if hasattr(cfg.model, "heads"):
        heads = nn.ModuleDict(
            {
                name: __head_factory__[name](**head_cfg, **kwargs)
                for name, head_cfg in cfg.model.heads.items() if head_cfg is not None
            }
        )
    else:
        heads = nn.ModuleDict()
    return heads


def build_postprocessor(cfg, **kwargs):
    __postprocessor_factory__ = {
        "hist_matching": HistogramMatching,
    }

    if hasattr(cfg.model, "postprocessors"):
        postprocessors = []
        for postprocessor_cfg in cfg.model.postprocessors:
            postprocessor_type = postprocessor_cfg.pop("type")
            postprocessor = __postprocessor_factory__[postprocessor_type](**postprocessor_cfg)
            postprocessors.append(postprocessor)
        postprocessor = Compose(*postprocessors)
    else:
        postprocessor = Compose()
    return postprocessor


def build_copy_detector(cfg, **kwargs):
    encoder = build_encoder(cfg, **kwargs)
    fusion = build_fusion(cfg, **kwargs)
    decoder = build_decoder(cfg, **kwargs)
    model = CopyDetector(encoder, fusion, decoder)

    return model


def build_paired_copy_detector(cfg, **kwargs):
    encoder = build_encoder(cfg, **kwargs)
    fusion = build_fusion(cfg, **kwargs)
    heads = build_heads(cfg, **kwargs)
    postprocessor = build_postprocessor(cfg, **kwargs)
    model = ClassificationCopyDetector(encoder, fusion, heads)
    model.register_postprocessor(postprocessor)

    return model


def build_gaze_copy_detector(cfg, **kwargs):
    encoder = build_encoder(cfg, **kwargs)
    fusion = build_fusion(cfg, **kwargs)
    heads = build_heads(cfg, **kwargs)
    model = GazeClassificationCopyDetector(encoder, fusion, heads)

    return model


def build_copy_descriptor(cfg, **kwargs):
    encoder = build_encoder(cfg, **kwargs)
    heads = build_heads(cfg, **kwargs)
    model = CopyDescriptor(encoder, heads)

    if hasattr(cfg.model, "freeze_encoder") and cfg.model.freeze_encoder:
        model.encoder.requires_grad_(False)
        logger.info(f"Encoder in model has been frozen and only heads are trainable.")

    return model


def build_copy_cnn_descriptor(cfg, **kwargs):
    encoder = build_cnn(cfg, **kwargs)
    heads = build_heads(cfg, **kwargs)
    model = CopyDescriptor(encoder, heads)

    return model


def build_local_verification_paired_copy_detector(cfg, **kwargs):
    encoder = build_encoder(cfg, **kwargs)
    fusion = build_fusion(cfg, **kwargs)
    heads = build_heads(cfg, **kwargs)
    model = LocalVerificationClassificationCopyDetector(encoder, fusion, heads)

    return model


def build_local_verification_copy_descriptor(cfg, **kwargs):
    encoder = build_encoder(cfg, **kwargs)
    heads = build_heads(cfg, **kwargs)
    model = LocalVerificationCopyDescriptor(encoder, heads)

    return model


def build_dino_model(cfg, **kwargs):
    model = build_encoder(cfg, last_norm=True, **kwargs)
    return model


def build_dinov2_model(cfg, **kwargs):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    logger.info(f"Only DINOv2 ViTs/14 is available.")
    return model


def build_dino_model_for_eval(cfg):
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            images = inputs["images"]
            return self.model(images)

    rank = distributed.get_local_rank()
    model = build_dino_model(cfg)

    ckpt = torch.load(dinos_pretrain_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    logger.info(f"=> Load parameters from {dinos_pretrain_path}.")

    model = Wrapper(model)
    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> DINO pretrained model built and ready for evaluation.")

    return model, 0


def build_dinov2_model_for_eval(cfg):
    rank = distributed.get_local_rank()
    model = build_dinov2_model(cfg)

    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> DINO pretrained model built and ready for evaluation.")

    return model


def build_model_for_eval(cfg):
    rank = distributed.get_local_rank()
    model = build_copy_detector(cfg)

    checkpointer = Checkpointer(
        rank,
        cfg.output_dir,
        model=model,
    )
    epoch = checkpointer.load(cfg.weights)
    logger.info(f"=> Load parameters from {cfg.weights}.")

    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> Model built and ready for evaluation.")

    return model, epoch


def build_encoder_from_model_for_eval(cfg):
    rank = distributed.get_local_rank()
    model = build_paired_copy_detector(cfg)

    checkpointer = Checkpointer(
        rank,
        cfg.output_dir,
        model=model,
    )
    epoch = checkpointer.load(cfg.weights)
    logger.info(f"=> Load parameters from {cfg.weights}.")

    model = model.encoder
    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> Model built and ready for evaluation.")

    return model, epoch


def build_paired_model_for_eval(cfg):
    rank = distributed.get_local_rank()
    model = build_paired_copy_detector(cfg)
    # model = build_gaze_copy_detector(cfg)

    checkpointer = Checkpointer(
        rank,
        cfg.output_dir,
        model=model,
    )
    epoch = checkpointer.load(cfg.weights)
    logger.info(f"=> Load parameters from {cfg.weights}.")

    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> Model built and ready for evaluation.")

    return model, epoch


def build_copy_descriptor_for_eval(cfg, **kwargs):
    rank = distributed.get_local_rank()
    model = build_copy_descriptor(cfg)

    checkpointer = Checkpointer(
        rank,
        cfg.output_dir,
        model=model,
    )
    epoch = checkpointer.load(cfg.weights)
    logger.info(f"=> Load parameters from {cfg.weights}.")

    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> Model built and ready for evaluation.")

    return model, epoch


def build_sscd_resnet50_descriptor_for_eval(cfg, **kwargs):
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            images = inputs["images"]
            return {
                "feat": self.model(images)
            }

    rank = distributed.get_local_rank()
    model_path = "sscd_disc_mixup.torchscript.pt"
    model = Wrapper(
        torch.jit.load(model_path)
    )

    epoch = 1
    logger.info(f"=> Load parameters from {model_path}.")

    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> SSCD R50 built and ready for evaluation.")

    return model, epoch


def build_resnet50_descriptor_for_eval(cfg, **kwargs):
    rank = distributed.get_local_rank()
    model = build_copy_cnn_descriptor(cfg)

    checkpointer = Checkpointer(
        rank,
        cfg.output_dir,
        model=model,
    )
    epoch = checkpointer.load(cfg.weights)
    logger.info(f"=> Load parameters from {cfg.weights}.")

    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> SSCD R50 built and ready for evaluation.")

    return model, epoch


def build_local_verification_paired_model_for_eval(cfg):
    rank = distributed.get_local_rank()
    model = build_local_verification_paired_copy_detector(cfg)

    checkpointer = Checkpointer(
        rank,
        cfg.output_dir,
        model=model,
    )
    epoch = checkpointer.load(cfg.weights)
    logger.info(f"=> Load parameters from {cfg.weights}.")

    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> Model built and ready for evaluation.")

    return model, epoch


def build_local_verification_copy_descriptor_for_eval(cfg):
    rank = distributed.get_local_rank()
    model = build_local_verification_copy_descriptor(cfg)

    checkpointer = Checkpointer(
        rank,
        cfg.output_dir,
        model=model,
    )
    epoch = checkpointer.load(cfg.weights)
    logger.info(f"=> Load parameters from {cfg.weights}.")

    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> Model built and ready for evaluation.")

    return model, epoch


def build_local_verification_dino_for_eval(cfg):
    rank = distributed.get_local_rank()
    model = build_dino_model(cfg)

    ckpt = torch.load(dinos_pretrain_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    logger.info(f"=> Load parameters from {dinos_pretrain_path}.")

    model = LocalVerificationDINO(model)
    model = model.to(torch.device(rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    logger.info("=> DINO pretrained model built and ready for evaluation.")

    return model, 0
