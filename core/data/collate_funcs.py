import torch
from torch.utils.data import default_collate


def np_collate_func(samples):
    assert isinstance(samples[0], dict) and "images" in samples[0] and "indices" in samples[0]

    images = torch.cat([sample.pop("images") for sample in samples], dim=0)
    indices = torch.cat([sample.pop("indices") for sample in samples], dim=0)
    samples = default_collate(samples)
    samples.update({"images": images, "indices": indices})

    return samples


def force_hard_negative_mining_collate_func(samples):
    samples_a, samples_b = default_collate(samples)
    samples = {}
    for key in samples_a.keys():
        samples[key] = torch.cat([samples_a[key], samples_b[key]], dim=0)

    return samples
