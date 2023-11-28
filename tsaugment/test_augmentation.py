import random
import unittest

import numpy as np
import pandas as pd

from typing import Union, Tuple, List, Dict, Any

from .augmentation import (
    add_white_noise,
    elementwise_amplitude_scaling,
    invert_signal_amplitude,
    random_scale,
    reverse_signal,
    random_shift,
    smooth_signal
)

ALWAYS = 1.1

class MyTestCase(unittest.TestCase):

    @staticmethod
    def reset_seed():
        SEED = 42

        np.random.random(SEED)
        random.seed(SEED)

    def case1(self) -> Tuple[Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]], Dict[str, Dict[str, Union[pd.Series, pd.DataFrame]]]]:
        signal = pd.Series(list(np.arange(10)) * 3)
        label = pd.DataFrame(
            ({"class": 0, "start": 9, "end": 10},
             {"class": 1, "start": 10, "end": 11},
             {"class": 2, "start": 29, "end": 30}
             ))

        ground_truth = dict()
        self.reset_seed()
        ground_truth["reverse_signal"] = {
            "signal": pd.Series(list(np.arange(9, -1, -1)) * 3),
            "label": pd.DataFrame(
            ({"class": 0, "start": 20, "end": 21},
             {"class": 1, "start": 19, "end": 20},
             {"class": 2, "start": 0, "end": 1}
             ))
        }
        self.reset_seed()
        ground_truth["invert_signal_amplitude"] = {
            "signal": pd.Series(list(np.arange(0, -10, -1)) * 3),
            "label": label
        }

        return (signal, label), ground_truth

    def case2(self) -> Tuple[Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]], Dict[str, Dict[str, Union[pd.Series, pd.DataFrame]]]]:
        signal = pd.Series([1] * 5 + [5] * 3 + [0] * 10 + [4] * 5)
        label = pd.DataFrame(
            ({"class": 42, "start": 5, "end": 8},
             {"class": 1, "start": 18, "end": 23}
             ))

        ground_truth = dict()
        # no randomness
        ground_truth["reverse_signal"] = {
            "signal": pd.Series([4] * 5 + [0] * 10 + [5] * 3 + [1] * 5),
            "label": pd.DataFrame(
                ({"class": 42, "start": 15, "end": 18},
                 {"class": 1, "start": 0, "end": 5}
                 ))
        }
        # no randomness
        ground_truth["invert_signal_amplitude"] = {
            "signal": pd.Series([-1] * 5 + [-5] * 3 + [0] * 10 + [-4] * 5),
            "label": label
        }

        self.reset_seed()

        return (signal, label), ground_truth

    def get_cases(self):
        return [self.case1(), self.case2()]

    def test_reverse_signal(self):
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = reverse_signal(signal, label, p=ALWAYS)
            # assert signal
            np.testing.assert_array_equal(sig, ground_truth["reverse_signal"]["signal"])
            # assert label
            np.testing.assert_array_equal(lbl, ground_truth["reverse_signal"]["label"])

    def test_invert_signal_amplitude(self):
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = invert_signal_amplitude(signal, label, p=ALWAYS)
            # assert signal
            np.testing.assert_array_equal(sig, ground_truth["invert_signal_amplitude"]["signal"])
            # assert label
            np.testing.assert_array_equal(lbl, ground_truth["invert_signal_amplitude"]["label"])

    def test_add_white_noise(self):
        scale = 0.05
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = add_white_noise(signal, label, amplitude_percent=scale, p=ALWAYS)
            # assert signal
            signal_max = np.max(signal) * (1 + scale * np.sign(np.max(signal)))
            signal_min = np.min(signal) * (1 - scale * np.sign(np.min(signal)))
            self.assertGreaterEqual(np.min(sig), signal_min)
            self.assertLessEqual(np.max(sig), signal_max)
            self.assertFalse(all([el1 != el2 for el1, el2 in zip(sig, signal)]))
            # assert label
            np.testing.assert_array_equal(lbl, label)

if __name__ == "__main__":
    unittest.main()
