"""
A set of utilities to manage and load checkpoints of training experiments.
"""

from pathlib import Path
import logging
import re
from omegaconf import OmegaConf
import torch

from ..settings import EXPER_PATH
from ..models import get_model


def list_checkpoints(dir_):
    """List all valid checkpoints in a given directory."""
    checkpoints = []
    for p in dir_.glob('checkpoint_*.tar'):
        numbers = re.findall(r'(\d+)', p.name)
        if len(numbers) == 0:
            continue
        assert len(numbers) == 1
        checkpoints.append((int(numbers[0]), p))
    return checkpoints


def get_last_checkpoint(exper, allow_interrupted=True):
    """Get the last saved checkpoint for a given experiment name."""
    ckpts = list_checkpoints(Path(EXPER_PATH, exper))
    if not allow_interrupted:
        ckpts = [(n, p) for (n, p) in ckpts if '_interrupted' not in p.name]
    assert len(ckpts) > 0
    return sorted(ckpts)[-1][1]


def get_best_checkpoint(exper):
    """Get the checkpoint with the best loss, for a given experiment name."""
    p = Path(EXPER_PATH, exper, 'checkpoint_best.tar')
    return p


def delete_old_checkpoints(dir_, num_keep):
    """Delete all but the num_keep last saved checkpoints."""
    ckpts = list_checkpoints(dir_)
    ckpts = sorted(ckpts)[::-1]
    kept = 0
    for ckpt in ckpts:
        if ('_interrupted' in str(ckpt[1]) and kept > 0) or kept >= num_keep:
            logging.info(f'Deleting checkpoint {ckpt[1].name}')
            ckpt[1].unlink()
        else:
            kept += 1


def load_experiment(exper, conf={}):
    """Load and return the model of a given experiment."""
    ckpt = get_best_checkpoint(exper)
    logging.info(f'Loading checkpoint {ckpt.name}')
    ckpt = torch.load(str(ckpt), map_location='cpu')

    conf = OmegaConf.merge(ckpt['conf'].model, OmegaConf.create(conf))
    model = get_model(conf.name)(conf).eval()
    model.load_state_dict(ckpt['model'])
    return model

def load_last_experiment(exper, conf={}):
    """Load and return the model of a given experiment."""
    ckpt = get_last_checkpoint(exper)
    logging.info(f'Loading checkpoint {ckpt.name}')
    ckpt = torch.load(str(ckpt), map_location='cpu')

    conf = OmegaConf.merge(ckpt['conf'].model, OmegaConf.create(conf))
    model = get_model(conf.name)(conf).eval()
    model.load_state_dict(ckpt['model'])
    return model

def flexible_load(state_dict, model):
    """TODO: fix a probable nasty bug, and move to BaseModel."""
    dict_params = set(state_dict.keys())
    model_params = set(map(lambda n: n[0], model.named_parameters()))

    if dict_params == model_params:  # prefect fit
        logging.info('Loading all parameters of the checkpoint.')
        model.load_state_dict(state_dict, strict=True)
        return
    elif len(dict_params & model_params) == 0:  # perfect mismatch
        strip_prefix = lambda x: '.'.join(x.split('.')[:1]+x.split('.')[2:])
        state_dict = {strip_prefix(n): p for n, p in state_dict.items()}
        dict_params = set(state_dict.keys())
        if len(dict_params & model_params) == 0:
            raise ValueError('Could not manage to load the checkpoint with'
                             'parameters:' + '\n\t'.join(sorted(dict_params)))
    common_params = dict_params & model_params
    left_params = dict_params - model_params
    logging.info('Loading parameters:\n\t'+'\n\t'.join(sorted(common_params)))
    if len(left_params) > 0:
        logging.info('Could not load parameters:\n\t'
                     + '\n\t'.join(sorted(left_params)))
    model.load_state_dict(state_dict, strict=False)
