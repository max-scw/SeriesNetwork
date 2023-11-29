import numpy as np
import pandas as pd
import random

from typing import Union, List, Tuple, Any

from .signal_processing import smooth, len_signal, shift_signal
from .utils import randpm1


def add_white_noise(
        signal: Union[np.ndarray, pd.DataFrame],
        label: pd.DataFrame,
        amplitude_percent: float = 0.02,
        p: float = 0.5
) -> Tuple[Union[np.ndarray, pd.DataFrame], pd.DataFrame]:
    if random.random() < p:  # chance
        amplitude = amplitude_percent * (np.max(signal) - np.min(signal))
        noise = randpm1() * amplitude
        signal += noise

    return signal, label


def elementwise_scaling(
        signal: Union[np.ndarray, pd.DataFrame],
        label: pd.DataFrame,
        amplitude_percent: float = 0.02,
        p: float = 0.5,
) -> Tuple[Union[np.ndarray, pd.DataFrame], pd.DataFrame]:
    if random.random() < p:  # chance
        noise = randpm1(signal.shape) * amplitude_percent
        signal *= 1 + noise

    return signal, label


def invert_signal_amplitude(
        signal: Union[np.ndarray, pd.DataFrame],
        label: pd.DataFrame,
        p: float = 0.5
) -> Tuple[Union[np.ndarray, pd.DataFrame], pd.DataFrame]:
    """mirrors the signal amplitude at 0, i.e. 10 => -10"""
    if random.random() < p:  # chance
        signal *= -1

    return signal, label


def reverse_signal(
        signal: Union[np.ndarray, pd.DataFrame],
        label: pd.DataFrame,
        keys: List[Any] = None,
        p: float = 0.5
) -> Tuple[Union[np.ndarray, pd.DataFrame], pd.DataFrame]:
    """reverses an array (aka. flips it column-wise)"""
    if random.random() < p:  # chance
        # reverse signal
        signal = signal[::-1]

        if isinstance(signal, (pd.Series, pd.DataFrame)):
            signal.index = signal.index[::-1]

        # adjust label
        keys = get_label_keys_to_adjust(keys, label)
        label[keys] = len_signal(signal) + 1 - label[keys]
        # flip if applies
        if all([el in keys for el in ["start", "end"]]):
            start = label["start"]
            label["start"] = label["end"]
            label["end"] = start

    return signal, label


def get_label_keys_to_adjust(keys: List[Any], label: pd.DataFrame) -> List[Any]:
    if keys is None:
        # potential keys
        keys_ = ["start", "end", "idx", "index", "x"]
        keys = [el for el in keys_ if el in label]
    return keys


def smooth_signal(
        signal: Union[np.ndarray, pd.DataFrame],
        label: pd.DataFrame,
        window: Union[str, List[str]] = "any",
        max_window_size: int = -1,
        p: float = 0.5
) -> Tuple[Union[np.ndarray, pd.DataFrame], pd.DataFrame]:
    """applies different smoothing filter to the signal. This is essentially a convolution."""
    if random.random() < p:  # chance
        # window size
        if max_window_size < 3:
            wz = random.randint(3, len(signal) // 80)
        else:
            wz = random.randint(3, max_window_size)

        # window type
        windows = ["flat", "hanning", "hamming", "bartlett", "blackman"]
        if isinstance(window, str):
            if window.lower() in windows:
                windows = [window]
        elif isinstance(window, list):
            windows = window
        # select active window type
        window = windows[random.randint(0, len(windows) - 1)]

        # smooth signal
        sig = smooth(signal, wz, window)

        # format output
        if isinstance(signal, pd.Series):
            signal = pd.Series(sig, index=signal.index, name=signal.name)
        elif isinstance(signal, pd.DataFrame):
            signal = pd.DataFrame(sig, index=signal.index, columns=signal.columns)

    return signal, label


def uniform_scale(
        signal: Union[np.ndarray, pd.DataFrame],
        label: pd.DataFrame,
        scale: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,
        always_apply: bool = False
) -> Tuple[Union[np.ndarray, pd.DataFrame], pd.DataFrame]:
    """random, uniform scaling of the signal amplitude"""
    if random.random() < p or always_apply:  # chance
        amplitude = (1 + (max(scale)/min(scale) - 1)) * randpm1()
        signal *= amplitude

    return signal, label


def random_shift(
        signal: Union[np.ndarray, pd.DataFrame],
        label: pd.DataFrame,
        keys: List[Any] = None,
        max_shift: Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]] = 0.1,
        wrap: bool = True,
        p: float = 0.5,
) -> Tuple[Union[np.ndarray, pd.DataFrame], pd.DataFrame]:
    if random.random() < p:  # chance
        sig, shift = shift_signal(signal, max_shift=max_shift, wrap=wrap)

        # adjust labels
        keys = get_label_keys_to_adjust(keys, label)
        label_new = label[keys] + shift

        if not wrap:
            if shift > 0:
                label_new = label_new[(label_new < len(sig)).all(axis=1)]
            elif shift < 0:
                label_new = label_new[(label_new > 0).all(axis=1)]
        else:
            label_new %= len(sig)


        # assign
        if np.all(label_new >= 0) and np.all(label_new <= len(sig)):
            # format output
            if isinstance(signal, pd.Series):
                signal = pd.Series(sig, index=signal.index, name=signal.name)
            elif isinstance(signal, pd.DataFrame):
                signal = pd.DataFrame(sig, index=signal.index, columns=signal.columns)
            label[keys] = label_new
            label = label.iloc[label_new.index, :]
        else:
            raise Exception

    return signal, label


def shift_sections():
    pass







# TODO: different noise (e.g. by frequency)
# TODO: jitter: move labels left/right
# TODO: shift entire signal
# TODO: mosaik => cut sections