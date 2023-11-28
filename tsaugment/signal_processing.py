import numpy as np
import pandas as pd
import random

from typing import Union, Tuple, List, Any

from .utils import randpm1

def len_signal(signal: Union[np.ndarray, pd.DataFrame]) -> int:
    if isinstance(signal, (pd.Series, pd.DataFrame)) and "int" in signal.index.dtype.name:
        return signal.index[-1]
    else:
        return len(signal)

def smooth(x, window_len: int = 11, window: str = "hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
    TODO: fix padding
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    window = window.lower()
    if window == "flat":  # moving average
        w = np.ones(window_len,'d')
    elif window == "hanning":
        w = np.hanning(window_len)
    elif window == "hamming":
        w = np.hamming(window_len)
    elif window == "bartlett":
        w = np.bartlett(window_len)
    elif window == "blackman":
        w = np.bartlett(window_len)
    else:
        raise ValueError(f"Unknown window type {window}. Should be on of {windows}.")

    s = np.concatenate((x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]))

    y = np.convolve(w / w.sum(), s, mode='same')
    return y[(window_len - 1):-(window_len - 1)]


def shift_signal(
        signal: Union[np.ndarray, pd.DataFrame],
        max_shift: Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]],
        wrap: bool = True
) -> Tuple[Union[np.ndarray, pd.DataFrame], int]:
    # process input
    if isinstance(max_shift, (int, float)):
        max_shift_ = (max_shift, max_shift)
    else:
        max_shift_ = tuple(max_shift)

    # positive values only
    max_shift_ = tuple(abs(el) for el in max_shift_)
    # scale if necessary
    max_shift_ = tuple(el * len_signal(signal) if el <= 1 else el for el in max_shift_)

    # shift
    if random.random() <= 0.5:
        # left
        max_shift = -max_shift_[0]
    else:
        # right
        max_shift = -max_shift_[1]
    shift = round(max_shift * randpm1())

    # shift index
    idx = np.mod(list(range(shift, len(signal) + shift)), len(signal))

    if not wrap:
        if shift > 0:
            for i, el in enumerate(idx):
                if el == 0:
                    break
                else:
                    idx[i] = 0
        else:
            for i in range(len(idx), 0, -1):
                if idx[i] == len(idx):
                    break
                else:
                    idx[i] = len(idx)

    return signal[idx], shift




if __name__ == "__main__":
    sig = np.arange(10)
    smooth(sig, 5, "hanning")