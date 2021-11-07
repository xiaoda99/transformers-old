from collections import defaultdict, OrderedDict, Counter
import numpy as np

def numpy(a, decimals=None):
    v = np.array(a) if isinstance(a, list) else a.detach().cpu().numpy()
    if decimals is not None: v = v.round(decimals)
    return v

def show_topk(values, indices, values_fn=lambda x: numpy(x, decimals=3), indices_fn=lambda x: x):
    return dict(OrderedDict(zip(indices_fn(indices), values_fn(values))))