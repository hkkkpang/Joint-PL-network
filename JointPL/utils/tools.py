"""
Various handy Python and PyTorch utils.
"""

import time
import inspect
import numpy as np
import os
import torch


class AverageMetric:
    def __init__(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, tensor):
        assert tensor.dim() == 1
        N = len(tensor)
        self._sum += tensor.mean().item() * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise ValueError(
                'Loss must have at least one example'
                'before it can be computed.')
        return self._sum / self._num_examples


class MedianMetric:
    def __init__(self):
        self._elements = []

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            raise ValueError(
                'Loss must have at least one example'
                'before it can be computed.')
        return np.nanmedian(self._elements)


def get_class(mod_name, base_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
       the module named mod_name, child of base_path.
    """
    mod_path = '{}.{}'.format(base_path, mod_name)
    mod = __import__(mod_path, fromlist=[''])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


class Timer(object):
    """A simpler timer context object.
    Usage:
    ```
    > with Timer('mytimer'):
    >   # some computations
    [mytimer] Elapsed: X
    ```
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.duration = time.time() - self.tstart
        if self.name is not None:
            print('[%s] Elapsed: %s' % (self.name, self.duration))


def set_num_threads(nt):
    """Force numpy and other libraries to use a limited number of threads."""
    try:
        import mkl
    except ImportError:
        pass
    else:
        mkl.set_num_threads(nt)
    torch.set_num_threads(1)
    os.environ['IPC_ENABLE'] = '1'
    for o in ['OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[o] = str(nt)
