"""Self-check for assign_guitars' neighbour smoothing.  Run in envs/tools: python test_assign_guitars.py"""
from assign_guitars import smooth_pans


def demo() -> None:
    """A lone outlier among close neighbours is pulled to their side; far-apart notes stay as they are."""
    run = [{"pitch": 60, "start": i * 0.1} for i in range(5)]
    assert all(p < 0 for p in smooth_pans(run, [-1, -1, 1, -1, -1]))
    far = [{"pitch": 60, "start": 0.0}, {"pitch": 60, "start": 10.0}]
    assert smooth_pans(far, [-1, 1]) == [-1, 1]
    print("ok")


if __name__ == "__main__":
    demo()
