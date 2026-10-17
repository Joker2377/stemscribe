"""Self-check for the two-guitar split score.  Run in envs/tools: python test_eval_guitar_split.py"""
from eval_guitar_split import score


def demo() -> None:
    """Perfect split -> 1 (either track order), everything on one track or on both -> 0.5, swapped half -> 0.5."""
    a = [{"pitch": 60 + i, "start": i * 1.0, "end": i + 0.5} for i in range(4)]
    b = [{"pitch": 40 + i, "start": i * 1.0, "end": i + 0.5} for i in range(4)]
    refs = [a, b]
    assert score(refs, a, b)["balanced_acc"] == 1.0
    assert score(refs, b, a)["balanced_acc"] == 1.0
    assert score(refs, a + b, [])["balanced_acc"] == 0.5
    assert score(refs, a + b, a + b)["balanced_acc"] == 0.5
    assert score(refs, a[:2] + b[:2], a[2:] + b[2:])["balanced_acc"] == 0.5
    assert score([a, [{"pitch": 1, "start": 99.0, "end": 99.5}]], a, a)["balanced_acc"] != 0.0  # one guitar heard -> NaN
    print("ok")


if __name__ == "__main__":
    demo()
