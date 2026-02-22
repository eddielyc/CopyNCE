
import numpy as np
from PIL import Image

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.models.model_utils import init_weights


class CopyDetector(nn.Module):
    def __init__(self, encoder, fusion, decoder):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.decoder = decoder

        self.classifier = nn.Linear(self.fusion.embed_dim, 2)

        logger.info(f"Build {self.__class__.__name__}.")

    def forward(self, input):
        queries, references = input["queries"], input["references"]
        que_encoder_output = self.encoder(queries)
        ref_encoder_output = self.encoder(references)

        que_tokens, ref_tokens = que_encoder_output["tokens"], ref_encoder_output["tokens"]
        tokens = torch.cat([que_tokens, ref_tokens], dim=1)

        fused_output = self.fusion(tokens)
        fused_que_tokens, fused_ref_tokens = fused_output["patch_tokens"].split(self.encoder.num_tokens, dim=1)
        fused_que_cls_token, fused_que_patch_tokens = self.encoder.split_tokens(fused_que_tokens)
        fused_ref_cls_token, fused_ref_patch_tokens = self.encoder.split_tokens(fused_ref_tokens)
        fused_cls_token = fused_output["cls_token"]
        preds = self.classifier(fused_cls_token)

        que_decoder_output = self.decoder(fused_que_patch_tokens)
        ref_decoder_output = self.decoder(fused_ref_patch_tokens)

        return {
            "que_masks": que_decoder_output["masks"],
            "ref_masks": ref_decoder_output["masks"],
            "preds": preds,
            "que_cls_token": que_encoder_output["cls_token"],
            "ref_cls_token": ref_encoder_output["cls_token"],

            "fused_cls_token": fused_cls_token,
            "fused_que_cls_token": fused_que_cls_token,
            "fused_ref_cls_token": fused_ref_cls_token,
            "fused_que_patch_tokens": fused_que_patch_tokens,
            "fused_ref_patch_tokens": fused_ref_patch_tokens,

            "que_encoder_output": que_encoder_output,
            "ref_encoder_output": ref_encoder_output,
            "fused_output": fused_output,
            "que_decoder_output": que_decoder_output,
            "ref_decoder_output": ref_decoder_output,
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'encoder.pos_embed', 'encoder.cls_token',
            'fusion.cls_token',
            'decoder.cls_emb',
        }


class DINOCopyDetector(nn.Module):
    def __init__(self, dino_encoder):
        super().__init__()
        self.encoder = dino_encoder

        logger.info(f"Build {self.__class__.__name__}.")

    def forward(self, input):
        queries, references = input["queries"], input["references"]
        que_encoder_output = self.encoder(queries)
        ref_encoder_output = self.encoder(references)
        que_cls_tokens = que_encoder_output["cls_token"]
        ref_cls_tokens = ref_encoder_output["cls_token"]
        que_cls_tokens = F.normalize(que_cls_tokens, dim=-1)
        ref_cls_tokens = F.normalize(ref_cls_tokens, dim=-1)

        preds = (que_cls_tokens * ref_cls_tokens).sum(dim=-1)

        return {
            "preds": preds,
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()


class ClassificationCopyDetector(nn.Module):
    def __init__(self, encoder, fusion, heads):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.heads = heads
        self.postprocessor = None

        logger.info(f"Build {self.__class__.__name__}.")

    def register_postprocessor(self, postprocessor):
        self.postprocessor = postprocessor
        logger.info(f"Register postprocessor for {self.__class__.__name__}.")

    def forward(self, input):
        queries, references = input["queries"], input["references"]
        que_encoder_output = self.encoder(queries)
        ref_encoder_output = self.encoder(references)

        que_tokens, ref_tokens = que_encoder_output["tokens"], ref_encoder_output["tokens"]
        tokens = torch.cat([que_tokens, ref_tokens], dim=1)

        fused_output = self.fusion(tokens)
        fused_que_tokens, fused_ref_tokens = fused_output["patch_tokens"].split(self.encoder.num_tokens, dim=1)
        fused_que_cls_token, fused_que_patch_tokens = self.encoder.split_tokens(fused_que_tokens)
        fused_ref_cls_token, fused_ref_patch_tokens = self.encoder.split_tokens(fused_ref_tokens)

        output = {
            "que_cls_token": que_encoder_output["cls_token"],
            "ref_cls_token": ref_encoder_output["cls_token"],

            "fused_cls_token": fused_output["cls_token"],
            "fused_que_cls_token": fused_que_cls_token,
            "fused_ref_cls_token": fused_ref_cls_token,
            "fused_que_patch_tokens": fused_que_patch_tokens,
            "fused_ref_patch_tokens": fused_ref_patch_tokens,

            "que_encoder_output": que_encoder_output,
            "ref_encoder_output": ref_encoder_output,
            "fused_output": fused_output,
        }

        output.update(self.forward_heads(output))
        if self.postprocessor is not None:
            output.update(self.postprocessor(output, input))
        return output

    def forward_heads(self, input):
        if not hasattr(self, "heads"):
            return dict()
        assert isinstance(self.heads, nn.ModuleDict), \
            f"Expect heads to be nn.ModuleDict, got {type(self.heads)} instead."
        heads_outputs = {}
        for head in self.heads.values():
            heads_outputs.update(head(input))
        return heads_outputs

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'encoder.pos_embed', 'encoder.cls_token',
            'fusion.cls_token',
        }


class NPCopyDetector(nn.Module):
    def __init__(self, encoder, fusion, miner):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.miner = miner

        self.classifier = nn.Linear(self.fusion.embed_dim, 1)

        logger.info(f"Build {self.__class__.__name__}.")

    def forward(self, input):
        images = input["images"]
        encoder_output = self.encoder(images)

        pairs_indices, cls_gt = self.miner(encoder_output, input)
        que_encoder_output = self.index_select_from_output(encoder_output, pairs_indices[:, 0])
        ref_encoder_output = self.index_select_from_output(encoder_output, pairs_indices[:, 1])

        que_tokens, ref_tokens = que_encoder_output["tokens"], ref_encoder_output["tokens"]
        tokens = torch.cat([que_tokens, ref_tokens], dim=1)

        fused_output = self.fusion(tokens)
        fused_que_tokens, fused_ref_tokens = fused_output["patch_tokens"].split(self.encoder.num_tokens, dim=1)
        fused_que_cls_token, fused_que_patch_tokens = self.encoder.split_tokens(fused_que_tokens)
        fused_ref_cls_token, fused_ref_patch_tokens = self.encoder.split_tokens(fused_ref_tokens)
        fused_cls_token = fused_output["cls_token"]
        preds = self.classifier(fused_cls_token)

        return {
            "preds": preds,
            "cls_gt": cls_gt,
            "cls_token": encoder_output["cls_token"],

            "fused_cls_token": fused_cls_token,
            "fused_que_cls_token": fused_que_cls_token,
            "fused_ref_cls_token": fused_ref_cls_token,
            "fused_que_patch_tokens": fused_que_patch_tokens,
            "fused_ref_patch_tokens": fused_ref_patch_tokens,

            "encoder_output": encoder_output,
            "fused_output": fused_output,
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'encoder.pos_embed', 'encoder.cls_token',
            'fusion.cls_token',
        }

    @staticmethod
    def index_select_from_output(output, indices):
        _output = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                value = value[indices]
            elif isinstance(value, (list, tuple)):
                value = [v[indices] for v in value]
            else:
                raise NotImplementedError
            _output[key] = value

        return _output


class GazeClassificationCopyDetector(ClassificationCopyDetector):
    def __init__(self, encoder, fusion, heads):
        super().__init__(encoder, fusion, heads)
        self.processor = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.40),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def forward(self, input):
        output = super().forward(input)

        batch_que_tokens = output["fused_que_patch_tokens"]
        batch_ref_tokens = output["fused_ref_patch_tokens"]

        gazed_queries, gazed_references = [], []
        for (
            query_ori, reference_ori,
            que_tokens, ref_tokens,
        ) in zip(
            input["queries_ori"], input["references_ori"],
            batch_que_tokens, batch_ref_tokens,
        ):
            query_ori = Image.fromarray(query_ori.cpu().numpy().astype(np.uint8))
            reference_ori = Image.fromarray(reference_ori.cpu().numpy().astype(np.uint8))
            gazed_query, gazed_reference = self.gaze_on_image(
                query_ori, reference_ori,
                que_tokens, ref_tokens,
                grid_size=(14, 14),
                matching_func=self.adaptively_match_copy_patch_indices,
                th=0.8, k2=3,
                min_n_patches=20,
            )
            gazed_queries.append(self.processor(gazed_query))
            gazed_references.append(self.processor(gazed_reference))
        gazed_queries = torch.stack(gazed_queries, dim=0)
        gazed_references = torch.stack(gazed_references, dim=0)
        input["queries"] = gazed_queries.cuda()
        input["references"] = gazed_references.cuda()

        output = super().forward(input)
        return output

    @staticmethod
    def _calc_entropy(aff, scale=5):
        logits = (aff * scale).exp()
        probs = logits / logits.sum(dim=1, keepdim=True)
        ent = (- probs * probs.log()).sum(dim=1)
        return ent

    def adaptive_topk_according_to_entropy(self, affinity, th=0.8, min_n_patches=25):
        ent_mat = self._calc_entropy(affinity, scale=5)
        abs_th = ent_mat.min() + (1 - th) * (ent_mat.max() - ent_mat.min())
        topk_a = torch.where(ent_mat < abs_th)[0].tolist()
        if len(topk_a) < min_n_patches:
            topk_a = torch.argsort(ent_mat)[:min_n_patches].cpu().numpy().tolist()

        ent_mat = self._calc_entropy(affinity.t(), scale=5)
        abs_th = ent_mat.min() + (1 - th) * (ent_mat.max() - ent_mat.min())
        topk_b = torch.where(ent_mat < abs_th)[0].tolist()
        if len(topk_b) < min_n_patches:
            topk_b = torch.argsort(ent_mat)[:min_n_patches].cpu().numpy().tolist()
        return topk_a, topk_b

    def adaptive_topk_related_tokens(self, tokens_a, tokens_b, th=0.8, min_n_patches=25):
        tokens_a = F.normalize(tokens_a, p=2, dim=1)
        tokens_b = F.normalize(tokens_b, p=2, dim=1)
        affinity = tokens_a @ tokens_b.t()
        topk_a, topk_b = self.adaptive_topk_according_to_entropy(affinity, th=th, min_n_patches=min_n_patches)

        return topk_a, topk_b

    def adaptively_match_copy_patch_indices(self, que_tokens, ref_tokens, th=0.8, k2=4, min_n_patches=25):
        que_tokens = F.normalize(que_tokens, p=2, dim=1)
        ref_tokens = F.normalize(ref_tokens, p=2, dim=1)
        affinity = que_tokens @ ref_tokens.t()
        que_topk, ref_topk = self.adaptive_topk_related_tokens(que_tokens, ref_tokens, th=th, min_n_patches=min_n_patches)

        # que to ref
        topk_affinity = affinity[que_topk]
        que_to_ref_knns = topk_affinity.argsort(descending=True)[:, :k2].flatten()
        que_to_ref_knns = torch.unique(que_to_ref_knns).tolist()

        # ref to que
        topk_affinity = affinity.t()[ref_topk]
        ref_to_que_knns = topk_affinity.argsort(descending=True)[:, :k2].flatten()
        ref_to_que_knns = torch.unique(ref_to_que_knns).tolist()

        # print(len(que_to_ref_knns), len(que_topk), len(ref_to_que_knns), len(ref_topk))
        if len(que_to_ref_knns) / len(que_topk) > len(ref_to_que_knns) / len(ref_topk):
            que_patch_indices, ref_patch_indices = que_topk, que_to_ref_knns
            # print(f"query to reference")
        else:
            que_patch_indices, ref_patch_indices = ref_to_que_knns, ref_topk
            # print(f"reference to query")

        return que_patch_indices, ref_patch_indices

    @staticmethod
    def gaze_on_image(
            query,
            reference,
            que_tokens,
            ref_tokens,
            grid_size,
            matching_func,
            *args,
            **kwargs
    ):
        grid_h, grid_w = grid_size
        que_patch_indices, ref_patch_indices = matching_func(que_tokens, ref_tokens, *args, **kwargs)
        que_patch_indices, ref_patch_indices = torch.tensor(que_patch_indices, dtype=torch.int), torch.tensor(
            ref_patch_indices, dtype=torch.int)

        # query
        que_patch_size_w, que_patch_size_h = query.width // grid_w, query.height // grid_h
        que_patch_row_indices = torch.div(que_patch_indices, grid_w, rounding_mode='floor')
        que_patch_col_indices = torch.remainder(que_patch_indices, grid_w)
        que_top = que_patch_row_indices.min() * que_patch_size_h
        que_bottom = que_patch_row_indices.max() * que_patch_size_h + que_patch_size_h
        que_left = que_patch_col_indices.min() * que_patch_size_w
        que_right = que_patch_col_indices.max() * que_patch_size_w + que_patch_size_w
        que_box = torch.tensor((que_left, que_top, que_right, que_bottom)).tolist()
        gazed_query = query.crop(que_box)

        # reference
        ref_patch_size_w, ref_patch_size_h = reference.width // grid_w, reference.height // grid_h
        ref_patch_row_indices = torch.div(ref_patch_indices, grid_w, rounding_mode='floor')
        ref_patch_col_indices = torch.remainder(ref_patch_indices, grid_w)
        ref_top = ref_patch_row_indices.min() * ref_patch_size_h
        ref_bottom = ref_patch_row_indices.max() * ref_patch_size_h + ref_patch_size_h
        ref_left = ref_patch_col_indices.min() * ref_patch_size_w
        ref_right = ref_patch_col_indices.max() * ref_patch_size_w + ref_patch_size_w
        ref_box = torch.tensor((ref_left, ref_top, ref_right, ref_bottom)).tolist()
        gazed_reference = reference.crop(ref_box)

        return gazed_query, gazed_reference


class CopyDescriptor(nn.Module):
    def __init__(self, encoder, heads):
        super().__init__()
        self.encoder = encoder
        self.heads = heads

        logger.info(f"Build {self.__class__.__name__}.")

    def forward(self, input):
        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_eval(input)

    def forward_train(self, input):
        queries, references = input["queries"], input["references"]
        que_encoder_output = self.encoder(queries)
        ref_encoder_output = self.encoder(references)

        output = {
            "que_cls_token": que_encoder_output["cls_token"],
            "ref_cls_token": ref_encoder_output["cls_token"],

            "que_patch_tokens": que_encoder_output["patch_tokens"],
            "ref_patch_tokens": ref_encoder_output["patch_tokens"],

            "que_encoder_output": que_encoder_output,
            "ref_encoder_output": ref_encoder_output,
        }

        output.update(self.forward_heads(output))
        return output

    def forward_eval(self, input):
        images = input["images"]
        encoder_output = self.encoder(images)

        output = {
            "cls_token": encoder_output["cls_token"],
            "encoder_output": encoder_output,
        }

        output.update(self.forward_heads(output))
        return output

    def forward_heads(self, input):
        if not hasattr(self, "heads"):
            return dict()
        assert isinstance(self.heads, nn.ModuleDict), \
            f"Expect heads to be nn.ModuleDict, got {type(self.heads)} instead."
        heads_outputs = {}
        for head in self.heads.values():
            heads_outputs.update(head(input))
        return heads_outputs

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'encoder.pos_embed', 'encoder.cls_token',
        }


class LocalVerificationClassificationCopyDetector(ClassificationCopyDetector):
    def forward(self, input):
        queries, references = input["queries"], input["references"]
        N, q_crops = queries.size(0), queries.size(1)
        r_crops = references.size(1)

        partial_preds = []
        for q_crops_i in range(q_crops):
            local_queries = queries[:, q_crops_i]
            global_references = references[:, 0]
            input["queries"], input["references"] = local_queries, global_references
            output = super().forward(input)
            partial_preds.append(output["preds"])

        for r_crops_i in range(1, r_crops):
            global_queries = queries[:, 0]
            local_references = references[:, r_crops_i]
            input["queries"], input["references"] = global_queries, local_references
            output = super().forward(input)
            partial_preds.append(output["preds"])

        preds = torch.stack(partial_preds, dim=0)
        assert preds.size(0) == q_crops + r_crops - 1
        preds, _ = preds.max(dim=0)
        output = {"preds": preds}

        return output


class LocalVerificationCopyDescriptor(CopyDescriptor):
    def forward_eval(self, input):
        images = input["images"]
        N, crops = images.size(0), images.size(1)

        images = images.view(N * crops, *images.shape[2:])
        encoder_output = self.encoder(images)

        output = {
            "cls_token": encoder_output["cls_token"],
        }
        output.update(self.forward_heads(output))
        output = self.recursive_unflatten_tensor_in_batch_wise(output, N, crops)
        return output

    def recursive_unflatten_tensor_in_batch_wise(self, value, n, crops):
        if isinstance(value, torch.Tensor):
            return value.view(n, crops, *value.shape[1:])
        elif isinstance(value, (list, tuple)):
            _value = []
            for v in value:
                _value.append(self.recursive_unflatten_tensor_in_batch_wise(v, n, crops))
            return _value
        elif isinstance(value, dict):
            _value = {}
            for k, v in value.items():
                _value[k] = self.recursive_unflatten_tensor_in_batch_wise(v, n, crops)
            return _value
        else:
            raise TypeError(f"Unsupported type for recursive_unflatten_tensor_in_batch_wise: {type(value)}")


class LocalVerificationDINO(LocalVerificationCopyDescriptor):
    def __init__(self, dino):
        super(LocalVerificationDINO, self).__init__(dino, None)

    def forward_eval(self, input):
        images = input["images"]
        N, crops = images.size(0), images.size(1)

        images = images.view(N * crops, *images.shape[2:])
        encoder_output = self.encoder(images)
        output = self.recursive_unflatten_tensor_in_batch_wise(encoder_output, N, crops)
        return output
