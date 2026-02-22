
from pathlib import Path
from loguru import logger

import torch
import torch.nn as nn
from torch.nn import Parameter


def save_checkpoint(state_dict, fpath='checkpoint.pth.tar'):
    Path(fpath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, fpath)


def load_weights(model, state_dict, key=None):
    if key is not None and key in state_dict:
        logger.info(f"Take key {key} in provided checkpoint dict")
        state_dict = state_dict[key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # add 'encoder.' prefix
    # state_dict = {"encoder." + k: v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loading weights for {'model' if key else key}... \n"
                f"missing keys: {missing_keys} \n"
                f"unexpected keys: {unexpected_keys} \n")


def load_checkpoint(fpath):
    fpath = Path(fpath)
    assert fpath.is_dir() or fpath.is_file(), 'previous checkpoint path not exists or not a folder'
    fpath = fpath / 'checkpoint.pth.tar' if fpath.is_dir() else fpath

    if fpath.is_file():
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        logger.info("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


class Checkpointer(object):
    def __init__(self, rank, logs_dir='./outputs', milestone_every=10, **modules):
        self.rank = rank
        self.logs_dir = Path(logs_dir)
        self.milestone_every = milestone_every
        self.modules = modules

    def step(self, epoch, fpath=None, **modules):
        if self.rank != 0:
            logger.info(f"=> Rank: {self.rank} will not save checkpoint.")
            return

        ckpt = {}
        modules.update(self.modules)
        for name, module in modules.items():
            if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                ckpt[name] = module.module.state_dict()
            else:
                ckpt[name] = module.state_dict()
        ckpt['epoch'] = epoch

        save_checkpoint(ckpt, self.logs_dir / f"checkpoint.pth.tar")

        if epoch % self.milestone_every == 0:
            fpath = self.logs_dir / f"checkpoint-epoch{epoch}.pth.tar" if fpath is None else fpath
            save_checkpoint(ckpt, fpath)
            logger.info(f"Save checkpoint of modules: {list(modules.keys())} into {fpath}.")

    def load(self, path=None):
        path = self.logs_dir if path is None else path
        try:
            state_dict = load_checkpoint(path)
        except (AssertionError, ValueError):
            logger.warning(f"No checkpoint found at {path} and will launch training from scratch"
                           f" or pretrained parameters.")
            return 0
        for name, module in self.modules.items():
            load_weights(module, state_dict, key=name)
        return state_dict.get("epoch", 0)
