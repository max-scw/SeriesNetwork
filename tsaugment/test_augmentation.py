import random
import unittest

import numpy as np
import pandas as pd

from typing import Union, Tuple, List, Dict, Any

from .augmentation import (
    add_white_noise,
    elementwise_scaling,
    invert_signal_amplitude,
    reverse_signal,
    random_shift,
    smooth_signal,
    uniform_scale,
)

ALWAYS = 1.1

class MyTestCase(unittest.TestCase):
    cases = None
    @staticmethod
    def reset_seed():
        SEED = 42

        np.random.seed(SEED)
        random.seed(SEED)

    def _case1(self) -> Tuple[Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]], Dict[str, Dict[str, Union[pd.Series, pd.DataFrame]]]]:
        signal = pd.Series(list(range(10)) * 3)
        label = pd.DataFrame(
            ({"class": 0, "start": 9, "end": 10},
             {"class": 1, "start": 19, "end": 20},
             {"class": 2, "start": 29, "end": 30}
             ))

        ground_truth = dict()
        self.reset_seed()
        ground_truth["reverse_signal"] = {
            "signal": pd.Series(list(np.arange(9, -1, -1)) * 3),
            "label": pd.DataFrame(
            ({"class": 0, "start": 20, "end": 21},
             {"class": 1, "start": 10, "end": 11},
             {"class": 2, "start": 0, "end": 1}
             ))
        }
        self.reset_seed()
        ground_truth["invert_signal_amplitude"] = {
            "signal": pd.Series(list(np.arange(0, -10, -1)) * 3),
            "label": label
        }

        # no randomness
        ground_truth["random_shift(max_shift=(4, 4), wrap=False)"] = {
            "signal": pd.Series([0] * 4 + list(np.arange(10)) * 2 + list(range(6))),
            "label": pd.DataFrame(
                ({"class": 0, "start": 13, "end": 14},
                 {"class": 1, "start": 23, "end": 24},
                 # {"class": 2, "start": 3, "end": 4}
                 ))
        }
        # left shift with wrap
        self.reset_seed()
        shift = np.random.randint(low=-4, high=0)
        ground_truth["random_shift(max_shift=(-4, 0), wrap=True)"] = {
            "signal": pd.Series(list(range(abs(shift), 10)) + list(range(10)) * 2 + list(range(abs(shift)))),
            "label": pd.DataFrame(
            ({"class": 0, "start": 9 + shift, "end": 10 + shift},
             {"class": 1, "start": 19 + shift, "end": 20 + shift},
             {"class": 2, "start": 29 + shift, "end": 30 + shift}
             ))
        }
        # right shift with wrap
        self.reset_seed()
        shift = np.random.randint(low=0, high=5)
        ground_truth["random_shift(max_shift=(0, 5), wrap=True)"] = {
            "signal": pd.Series(list(range(10 - shift, 10)) + list(range(10)) * 2 + list(range(10 - shift))),
            "label": pd.DataFrame(
            ({"class": 0, "start": 9 + shift, "end": 10 + shift},
             {"class": 1, "start": 19 + shift, "end": 20 + shift},
             {"class": 2, "start": (29 + shift) % len(signal), "end": (30 + shift) % len(signal)}
             ))
        }

        return (signal, label), ground_truth

    def _case2(self) -> Tuple[Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]], Dict[str, Dict[str, Union[pd.Series, pd.DataFrame]]]]:
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
        # no randomness
        ground_truth["random_shift(max_shift=(4, 4), wrap=False)"] = {
            "signal": pd.Series([1] * (5 + 4) + [5] * 3 + [0] * 10 + [4] * (5 - 4)),
            "label": pd.DataFrame(
                ({"class": 42, "start": 9, "end": 12},
                 # {"class": 1, "start": 22, "end": 4}
                 ))
        }
        # left shift with wrap
        self.reset_seed()
        shift = np.random.randint(low=-4, high=0)
        ground_truth["random_shift(max_shift=(-4, 0), wrap=True)"] = {
            "signal": pd.Series([1] * (5 + shift) + [5] * 3 + [0] * 10 + [4] * 5 + [1] * abs(shift)),
            "label": pd.DataFrame(
                ({"class": 42, "start": 5 + shift, "end": 8 + shift},
                 {"class": 1, "start": 18 + shift, "end": 23 + shift}
                 ))
        }
        # right shift with wrap
        self.reset_seed()
        shift = np.random.randint(low=0, high=5)
        print(f"DEBUG: [_case2()] shift={shift}")
        ground_truth["random_shift(max_shift=(0, 5), wrap=True)"] = {
            "signal": pd.Series([4] * shift + [1] * 5 + [5] * 3 + [0] * 10 + [4] * (5 - shift)),
            "label": pd.DataFrame(
                ({"class": 42, "start": 5 + shift, "end": 8 + shift},
                 {"class": 1, "start": (18 + shift) % len(signal), "end": (23 + shift) % len(signal)}
                 ))
        }

        self.reset_seed()

        return (signal, label), ground_truth

    def get_cases(self) -> List[tuple]:
        if self.cases is None:
            self.cases = [self._case1(), self._case2()]
        return self.cases

    def test_reverse_signal(self):
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = reverse_signal(signal, label, p=ALWAYS)
            # assert signal
            key = "reverse_signal"
            np.testing.assert_array_equal(sig, ground_truth[key]["signal"])
            # assert label
            np.testing.assert_array_equal(lbl, ground_truth[key]["label"])

    def test_invert_signal_amplitude(self):
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = invert_signal_amplitude(signal, label, p=ALWAYS)
            # assert signal
            key = "invert_signal_amplitude"
            np.testing.assert_array_equal(sig, ground_truth[key]["signal"])
            # assert label
            np.testing.assert_array_equal(lbl, ground_truth[key]["label"])

    def test_add_white_noise_range(self):
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

    def test_elementwise_scaling_range(self):
        scale = 0.15
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = elementwise_scaling(signal, label, amplitude_percent=scale, p=ALWAYS)
            # assert signal
            signal_max = np.max(signal) * (1 + scale * np.sign(np.max(signal)))
            signal_min = np.min(signal) * (1 - scale * np.sign(np.min(signal)))
            self.assertGreaterEqual(np.min(sig), signal_min)
            self.assertLessEqual(np.max(sig), signal_max)
            self.assertFalse(all([el1 != el2 for el1, el2 in zip(sig, signal)]))
            # assert label
            np.testing.assert_array_equal(lbl, label)

    def test_uniform_scale_range(self):
        scale = (0.9, 1.5)
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = uniform_scale(signal, label, scale, p=ALWAYS)
            # assert signal
            signal_max = np.max(np.max(signal) * np.asarray(scale))
            signal_min = np.min(np.min(signal) * np.asarray(scale))
            self.assertGreaterEqual(np.min(sig), signal_min)
            self.assertLessEqual(np.max(sig), signal_max)
            self.assertFalse(all([el1 != el2 for el1, el2 in zip(sig, signal)]))
            # assert label
            np.testing.assert_array_equal(lbl, label)

    def test_smooth_signal_range(self):
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = smooth_signal(signal, label, max_window_size=5, p=ALWAYS)
            # assert signal
            self.assertGreaterEqual(np.min(sig), np.min(signal))
            self.assertLessEqual(np.max(sig), np.max(signal))
            # assert label
            np.testing.assert_array_equal(lbl, label)

    def test_random_shift_range(self):
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = random_shift(signal, label, max_shift=4, p=ALWAYS)
            # assert signal
            self.assertEqual(np.min(sig), np.min(signal))
            self.assertEqual(np.max(sig), np.max(signal))
            # assert label
            self.assertGreaterEqual(np.min(lbl.to_numpy()), 0)
            self.assertGreaterEqual(np.max(lbl.to_numpy()), np.max(signal))

    def test_random_shift_1(self):
        for (signal, label), ground_truth in self.get_cases():
            # apply function
            sig, lbl = random_shift(signal, label, max_shift=(4, 4), wrap=False, p=ALWAYS)
            # assert signal
            key = "random_shift(max_shift=(4, 4), wrap=False)"
            np.testing.assert_array_equal(sig, ground_truth[key]["signal"])
            # assert label
            np.testing.assert_array_equal(lbl, ground_truth[key]["label"])

    def test_random_shift_2(self):
        cases = [
            (-4, 0),
            # (0, 5)
        ]

        for shift_range in cases:
            for (signal, label), ground_truth in self.get_cases():
                self.reset_seed()
                # apply function
                sig, lbl = random_shift(signal, label, max_shift=shift_range, wrap=True, p=ALWAYS)
                # assert signal
                key = f"random_shift(max_shift={shift_range}, wrap=True)"
                np.testing.assert_array_equal(sig, ground_truth[key]["signal"])
                # assert label
                np.testing.assert_array_equal(lbl, ground_truth[key]["label"])

if __name__ == "__main__":
    unittest.main()
