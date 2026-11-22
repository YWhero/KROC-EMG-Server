import unittest

import numpy as np

from kroc_act.emg import EMGStream, rms


class EMGTests(unittest.TestCase):
    def stream(self):
        t = np.arange(1000, dtype=np.int64) * 500000
        return EMGStream(t, t.copy(), np.tile(np.arange(1000)[:, None], (1, 4)), np.ones(1000, bool))

    def test_future_samples_never_enter_window(self):
        s = self.stream()
        now = 300000000
        expected = s.window(now, 400, 2000)
        s.values[s.time_ns > now] = 1e9
        np.testing.assert_array_equal(s.window(now, 400, 2000), expected)

    def test_availability_blocks_delayed_sample(self):
        s = self.stream()
        s.available_ns[-220:] += 1000000000
        with self.assertRaises(ValueError):
            s.window(s.time_ns[-1], 400, 2000)

    def test_window_ends_at_latest_received_sample(self):
        s = self.stream()
        s.available_ns[-10:] += 1000000000
        window = s.window(s.time_ns[-1], 400, 2000)
        self.assertEqual(window[-1, 0], 989)

    def test_missing_history_and_gap_rejected(self):
        s = self.stream()
        with self.assertRaises(ValueError):
            s.window(1000000, 400, 2000)
        s.valid[400:500] = False
        with self.assertRaises(ValueError):
            s.window(300000000, 400, 2000)

    def test_rms_uses_only_trailing_interval(self):
        x = np.full((400, 4), 1000, np.float32)
        x[-100:] = 3
        np.testing.assert_allclose(rms(x, 100), [3]*4)

    def test_numerically_exact_target_times(self):
        x = self.stream().window(300000000, 400, 2000)
        self.assertEqual(x.shape, (400, 4))
        np.testing.assert_array_equal(x[:, 0], np.arange(201, 601))


if __name__ == "__main__":
    unittest.main()
