from .copy_dataset import (
    CopyDataset,
    CopyDatasetMatchingEval,
    UnpairedCopyDataset,
    NPCopyDataset,
    LocalVerificationCopyDatasetMatchingEval,
    LocalVerificationCopyDatasetDescriptorEval,
    CopyDatasetDescriptorEval,
    PairedCopyDataset,
    SupervisedCopyDataset,
)

from .retrieval_dataset import OxfordParisDataset


class JointDataset(object):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.total = sum([len(d) for d in self.datasets])

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            else:
                index -= len(dataset)
        raise IndexError(index)
