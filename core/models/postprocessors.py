import math
import torch
from torch.nn import functional as F


class PostProcessor(object):
    def __call__(self, outputs, inputs):
        raise NotImplementedError


class HistogramMatching(PostProcessor):
    def __init__(self, anchor_hists_path, weight):
        self.anchor_hists = torch.load(anchor_hists_path)
        self.weight = weight

    def __call__(self, outputs, inputs):
        preds = outputs['preds']
        # fusion_head_que_patch_tokens: [B, N, C]
        fusion_head_que_patch_tokens = outputs['fusion_head_que_patch_tokens']
        fusion_head_que_patch_tokens = F.normalize(fusion_head_que_patch_tokens, p=2, dim=-1)
        # fusion_head_ref_patch_tokens: [B, N, C]
        fusion_head_ref_patch_tokens = outputs['fusion_head_ref_patch_tokens']
        fusion_head_ref_patch_tokens = F.normalize(fusion_head_ref_patch_tokens, p=2, dim=-1)
        # affinities: [B, N, N]
        affinities = torch.bmm(fusion_head_que_patch_tokens, fusion_head_ref_patch_tokens.permute((0, 2, 1)))
        hists = []
        for affinity in affinities:
            n, bins = torch.histogram(affinity.detach().view(-1).cpu(), bins=200, range=(-1., 1.))
            hists.append(n / n.sum())
        hists = torch.stack(hists, dim=0)
        kl_divs = self.kl_divergence(hists).to(preds.device)
        outputs.update(
            {
                "histograms": hists.to(preds.device),
                "cls_preds": preds,
                "kl_divs": kl_divs,
                "preds": (1. - self.weight) * preds - self.weight * kl_divs,
            }
        )
        return outputs

    def kl_divergence(self, hists, eps=1e-9, base=math.e):
        log_base = torch.log(torch.tensor(base)).item()
        assert len(hists.size()) == 2
        hists = hists + eps
        anchor_hists = self.anchor_hists + eps
        hists = hists / hists.sum(dim=-1, keepdims=True)
        anchor_hists = anchor_hists / anchor_hists.sum(dim=-1, keepdims=True)
        return ((-hists.log() / log_base) @ anchor_hists.t()).min(dim=-1)[0]


class Compose(PostProcessor):
    def __init__(self, *postprocessors):
        self.postprocessors = postprocessors

    def __call__(self, outputs, inputs):
        for postprocessor in self.postprocessors:
            outputs = postprocessor(outputs, inputs)
        return outputs
