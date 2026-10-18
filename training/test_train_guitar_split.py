"""Self-check for the guitar-split loss, role order and augmentation.  Run in envs/separator: python training/test_train_guitar_split.py"""
import random

import numpy as np
import torch

from train_guitar_split import SEG, augment, build_index, fixed_loss, mean_pitch, sample_example, thresholded_snr


def demo() -> None:
    """Silent refs stay finite, the loss cares about output order, and real examples come out
    as a finite (SEG,) mixture with two (SEG,) targets."""
    torch.manual_seed(0)
    ref = torch.randn(3, 2, 1, 1000)
    ref[0, 1] = 0  # one guitar silent
    mix = ref.sum(1)
    assert torch.isfinite(thresholded_snr(torch.zeros_like(mix), ref[:, 1], mix)).all()
    assert fixed_loss(ref, ref, mix) < fixed_loss(ref.flip(1), ref, mix)
    assert fixed_loss(ref, ref, mix) < fixed_loss(torch.zeros_like(ref), ref, mix)

    patches = build_index("validation")[0]
    files = [f for fs in patches.values() for f in fs]
    assert 20 < mean_pitch(files) < 100, "MIDI not found or wrong pitch"
    rng = random.Random(0)
    ex = next(e for e in (sample_example(patches, rng, need_both=True) for _ in range(20)) if e is not None)
    for _ in range(5):
        m, y = augment(ex, rng)
        assert m.shape == (SEG,) and y.shape == (2, SEG) and np.isfinite(m).all() and np.isfinite(y).all()
        assert np.abs(y).max() > 0
    print("ok")


if __name__ == "__main__":
    demo()
