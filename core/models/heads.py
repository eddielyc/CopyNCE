
from loguru import logger
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

from .fusion import Fusion
from .model_utils import init_weights
from ..layers import Block, Attention, Mlp


class ClassificationHead(Fusion):
    def __init__(
            self,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            block_layers=Block,
            act_layer=nn.GELU,
            attention_block=Attention,
            mlp_block=Mlp,
            init_scale=1e-4,
            cls_token=False,
            **kwargs
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            block_layers=block_layers,
            act_layer=act_layer,
            attention_block=attention_block,
            mlp_block=mlp_block,
            init_scale=init_scale,
            cls_token=cls_token,
            **kwargs
        )

        self.depth = depth
        self.add_cls_token = cls_token
        self.classifier = nn.Linear(embed_dim, 1)
        self.classifier.apply(init_weights)
        logger.info(self)

    def __repr__(self):
        return f"Build {self.__class__.__name__} with depth: {self.depth}, add_cls_token: {self.add_cls_token}."

    def get_num_layers(self):
        return len(self.blocks) + 1

    def split_tokens(self, tokens):
        cls_tokens, patch_tokens = tokens[:, 0], tokens[:, 1:]
        return cls_tokens, patch_tokens

    def prepare_input(self, x):
        if self.cls_token is not None:
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            if x.size(1) % 2 == 1:
                # has original class token and drop it
                x = x[:, 1:]
            x = torch.cat((cls_tokens, x), dim=1)
        return x

    def forward(self, input):
        x = input["fused_output"]["tokens"]
        x = self.prepare_input(x)

        block_outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            block_outputs.append(x)
        x = self.norm(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)

        cls_token, patch_tokens = self.split_tokens(x)

        preds = self.classifier(cls_token)
        if not self.training:
            preds = torch.sigmoid(preds)

        return {
            "cls_head_block_outputs": block_outputs,
            "cls_head_tokens": x,
            "cls_head_cls_token": cls_token,
            "cls_head_patch_tokens": patch_tokens,
            "preds": preds,
        }


class GazeClassificationHead(ClassificationHead):
    def __init__(
            self,
            k=98,
            gaze_type='gaze',

            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            block_layers=Block,
            act_layer=nn.GELU,
            attention_block=Attention,
            mlp_block=Mlp,
            init_scale=1e-4,
            cls_token=False,
            **kwargs
    ):
        __gaze_factory__ = {
            "entropy": self._gaze_according_to_entropy,
        }

        self.k = k
        self.gaze_type = gaze_type
        self.gaze_func = __gaze_factory__[gaze_type]

        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            block_layers=block_layers,
            act_layer=act_layer,
            attention_block=attention_block,
            mlp_block=mlp_block,
            init_scale=init_scale,
            cls_token=cls_token,
            **kwargs
        )

    def __repr__(self):
        return f"Build {self.__class__.__name__} with depth: {self.depth}, k: {self.k}, gaze_type: {self.gaze_type}, " \
               f"add_cls_token: {self.add_cls_token}."

    @staticmethod
    def _calc_entropy(batch_aff, scale=5):
        assert len(batch_aff.size()) == 3
        logits = (batch_aff * scale).exp()
        probs = logits / logits.sum(dim=2, keepdim=True)
        ent = (- probs * probs.log()).sum(dim=2)
        return ent

    def _gaze_according_to_entropy(self, batch_affinity):

        ent_mat = self._calc_entropy(batch_affinity, scale=5)
        topk_a = torch.argsort(ent_mat, dim=1)

        ent_mat = self._calc_entropy(batch_affinity.transpose(1, 2), scale=5)
        topk_b = torch.argsort(ent_mat, dim=1)
        return topk_a[:, : self.k], topk_b[:, : self.k]

    def gaze(self, que_tokens, ref_tokens):
        que_tokens = F.normalize(que_tokens, p=2, dim=2)
        ref_tokens = F.normalize(ref_tokens, p=2, dim=2)
        batch_affinity = torch.bmm(que_tokens, ref_tokens.transpose(1, 2))

        batch_que_indices, batch_ref_indices = self.gaze_func(batch_affinity)

        marked_que_tokens = torch.stack(
            [
                _que_tokens[que_indices]
                for que_indices, _que_tokens in zip(batch_que_indices, que_tokens)
            ], dim=0
        )
        marked_ref_tokens = torch.stack(
            [
                _ref_tokens[ref_indices]
                for ref_indices, _ref_tokens in zip(batch_ref_indices, ref_tokens)
            ], dim=0
        )
        return marked_que_tokens, marked_ref_tokens

    def forward(self, input):
        fused_cls_token = input["fused_cls_token"]
        fused_que_patch_tokens = input["fused_que_patch_tokens"]
        fused_ref_patch_tokens = input["fused_ref_patch_tokens"]

        marked_que_tokens, marked_ref_tokens = self.gaze(fused_que_patch_tokens, fused_ref_patch_tokens)
        x = torch.cat([fused_cls_token.unsqueeze(dim=1), marked_que_tokens, marked_ref_tokens], dim=1)
        x = self.prepare_input(x)

        block_outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            block_outputs.append(x)
        x = self.norm(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)

        cls_token, patch_tokens = self.split_tokens(x)

        preds = self.classifier(cls_token)

        return {
            "gaze_cls_head_block_outputs": block_outputs,
            "gaze_cls_head_tokens": x,
            "gaze_cls_head_cls_token": cls_token,
            "gaze_cls_head_patch_tokens": patch_tokens,
            "preds": preds,
        }


class FusionHead(Fusion):
    def __init__(
            self,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            block_layers=Block,
            act_layer=nn.GELU,
            attention_block=Attention,
            mlp_block=Mlp,
            init_scale=1e-4,
            cls_token=False,
            **kwargs
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            block_layers=block_layers,
            act_layer=act_layer,
            attention_block=attention_block,
            mlp_block=mlp_block,
            init_scale=init_scale,
            cls_token=cls_token,
            **kwargs
        )
        self.input_mode = kwargs["input_mode"]

        logger.info(f"Build {self.__class__.__name__} with depth: {depth}, input_mode: {self.input_mode}.")

    def fetch_tokens(self, input):
        if self.input_mode == "fusion":
            que_patch_tokens = input["fused_que_patch_tokens"]
            ref_patch_tokens = input["fused_ref_patch_tokens"]
        elif self.input_mode == "descriptor":
            que_patch_tokens = input["que_patch_tokens"]
            ref_patch_tokens = input["ref_patch_tokens"]
        else:
            raise ValueError(f"Invalid input mode: {self.input_mode}.")

        return que_patch_tokens, ref_patch_tokens

    def forward(self, input):
        que_patch_tokens, ref_patch_tokens = self.fetch_tokens(input)

        x = torch.cat(
            [
                que_patch_tokens,
                ref_patch_tokens,
            ], dim=1
        )
        x = self.prepare_input(x)

        block_outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            block_outputs.append(x)
        x = self.norm(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)

        cls_token, patch_tokens = self.split_tokens(x)
        fusion_head_que_patch_tokens = patch_tokens[:, : que_patch_tokens.size(1)]
        fusion_head_ref_patch_tokens = patch_tokens[:, que_patch_tokens.size(1):]

        return {
            "fusion_head_cls_token": cls_token,
            "fusion_head_que_patch_tokens": fusion_head_que_patch_tokens,
            "fusion_head_ref_patch_tokens": fusion_head_ref_patch_tokens,
        }


class ProjectionHead(nn.Module):
    def __init__(self, encoder_embed_dim, embed_dim, fc_dim):
        super().__init__()
        self.fc1 = nn.Linear(encoder_embed_dim, embed_dim, bias=False)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.fc2 = nn.Linear(embed_dim, fc_dim, bias=False)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward_proj(self, x):
        embeddings = self.fc1(x)
        projecteds = self.bn(embeddings)
        projecteds = self.fc2(projecteds)
        projecteds = F.normalize(projecteds, dim=1, p=2)

        return embeddings, projecteds

    def forward(self, input):
        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_eval(input)

    def forward_train(self, input):
        assert self.training
        xq, xr = input['que_cls_token'], input['ref_cls_token']
        embeddings_q, projecteds_q = self.forward_proj(xq)
        embeddings_r, projecteds_r = self.forward_proj(xr)

        return {
            "que_embedding": embeddings_q,
            "ref_embedding": embeddings_r,
            "que_feat": projecteds_q,
            "ref_feat": projecteds_r
        }

    def forward_eval(self, input):
        assert not self.training
        x = input['cls_token']
        embeddings, projecteds = self.forward_proj(x)

        return {
            "embedding": embeddings,
            "feat": projecteds,
        }


class LinearHead(nn.Module):
    def __init__(self, embed_dim, fc_dim):
        super().__init__()
        self.fc_dim = fc_dim
        self.fc1 = nn.Linear(embed_dim, fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward_linear(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = F.normalize(x, dim=1, p=2)

        return x

    def forward(self, input):
        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_eval(input)

    def forward_train(self, input):
        assert self.training
        xq, xr = input['que_cls_token'], input['ref_cls_token']
        xq = self.forward_linear(xq)
        xr = self.forward_linear(xr)

        return {
            "que_feat": xq,
            "ref_feat": xr
        }

    def forward_eval(self, input):
        assert not self.training
        x = input['cls_token']
        x = self.forward_linear(x)

        return {
            "feat": x,
        }
