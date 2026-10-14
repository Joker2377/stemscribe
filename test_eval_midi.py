"""Sanity check for eval_midi scoring: exact copy scores 1, a 100 ms shift scores 0."""
from eval_midi import score_drums, score_pitched

ref = [{"pitch": 60 + i % 5, "start": i * 0.5, "end": i * 0.5 + 0.3} for i in range(20)]
shifted = [{**n, "start": n["start"] + 0.1, "end": n["end"] + 0.1} for n in ref]
assert score_pitched(ref, ref)["f1"] == 1.0 and score_pitched(ref, ref)["f1_offset"] == 1.0
assert score_pitched(ref, shifted)["f1"] == 0.0
assert score_drums(ref, ref)["f1"] == 1.0 and score_drums(ref, shifted)["f1"] == 0.0
print("ok")

from eval_midi import score_pitched_octave
up = [{**n, "pitch": n["pitch"] + 12} for n in ref]
assert score_pitched_octave(up, ref)["f1"] == 1.0 and score_pitched_octave(up, ref)["octave_fix"] == -12
print("ok octave")
