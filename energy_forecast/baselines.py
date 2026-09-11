"""模块用途：仅使用起点之前的原单位历史进行朴素预测。"""
import numpy as np
from .data import finite, positive_int


def predict(method, history, horizon):
    positive_int(horizon,'horizon')
    history = np.asarray(history,dtype=np.float64)
    if history.ndim != 2 or not len(history) or history.shape[1] == 0:
        raise ValueError('history must be a nonempty [hours, targets] array')
    finite(history,'baseline history')
    if method == 'persistence':
        return np.repeat(history[-1:],horizon,axis=0)
    if method == 'seasonal24':
        if len(history) < 24: raise ValueError('seasonal24 requires at least 24 historical hours')
        return history[-24:][np.arange(horizon)%24]
    raise ValueError(f'Unknown baseline: {method}')
