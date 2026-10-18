"""Stage 2.6: strip reverb tails from pitched stems before restoration / transcription.

Runs in envs/separator.  Usage: dereverb.py <work_dir> <stem_variant>
Writes 01_stems/<variant>_dry/: dry versions of the stems in MODELS, the rest hard-linked.
vocals.wav is first split by a karaoke model into lead_vocals (dereverbed) + backing_vocals (kept wet,
because the vocal dereverb treats harmonies as reverb and removes them).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import soundfile as sf
from audio_separator.separator import Separator

from separate_uvr import MODEL_DIR, stem_name

# stem -> dereverb model; anvuew's roformers are vocal-only, MDX23C also works on instruments
MODELS = {
    "vocals": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "guitar": "MDX23C-De-Reverb-aufr33-jarredou.ckpt",
    "piano": "MDX23C-De-Reverb-aufr33-jarredou.ckpt",
}
DRY_NAMES = {"dry", "noreverb"}
KARAOKE = "mel_band_roformer_karaoke_becruily.ckpt"  # outputs (Vocals)=lead, (Instrumental)=backing


def run_model(wav: Path, model_file: str) -> dict:
    """Run one audio-separator model on a wav; return {output stem name: (audio, sr)}."""
    with tempfile.TemporaryDirectory() as tmp:
        # ponytail: reloads the model per call (a few calls, a few s each); cache if stems grow
        sep = Separator(model_file_dir=str(MODEL_DIR), output_dir=tmp, normalization_threshold=1.0)
        sep.load_model(model_file)
        files = [Path(tmp) / Path(f).name for f in sep.separate(str(wav))]
        return {stem_name(f): sf.read(f, dtype="float32") for f in files}


def dry_version(wav: Path, model_file: str) -> tuple:
    """Run one dereverb model on a stem; return (dry audio, sr)."""
    return next(v for k, v in run_model(wav, model_file).items() if k in DRY_NAMES)


def split_vocals(wav: Path, dst: Path) -> None:
    """Karaoke-split vocals.wav into dst/lead_vocals.wav (dereverbed) and dst/backing_vocals.wav (wet)."""
    parts = run_model(wav, KARAOKE)
    sf.write(dst / "backing_vocals.wav", *parts["instrumental"], subtype="FLOAT")
    with tempfile.TemporaryDirectory() as tmp:
        lead = Path(tmp) / "lead.wav"
        sf.write(lead, *parts["vocals"], subtype="FLOAT")
        sf.write(dst / "lead_vocals.wav", *dry_version(lead, MODELS["vocals"]), subtype="FLOAT")


def main(work: Path, variant: str) -> None:
    """Dereverb every stem listed in MODELS; hard-link the others."""
    src, dst = work / "01_stems" / variant, work / "01_stems" / f"{variant}_dry"
    dst.mkdir(parents=True, exist_ok=True)
    for wav in src.glob("*.wav"):
        if wav.stem == "vocals":
            split_vocals(wav, dst)
            print(f"vocals: karaoke {KARAOKE} -> lead (dry) + backing (wet)", flush=True)
            continue
        model_file = MODELS.get(wav.stem)
        if model_file is None:
            if not (dst / wav.name).exists():
                os.link(wav, dst / wav.name)
            continue
        audio, sr = dry_version(wav, model_file)
        sf.write(dst / wav.name, audio, sr, subtype="FLOAT")
        print(f"{wav.stem}: dry via {model_file}", flush=True)
    meta = json.loads((src / "stems.json").read_text())
    (dst / "stems.json").write_text(json.dumps({**meta, "dereverb": MODELS, "karaoke": KARAOKE}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
