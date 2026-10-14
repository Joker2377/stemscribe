"""Stage 2.6: strip reverb tails from pitched stems before restoration / transcription.

Runs in envs/separator.  Usage: dereverb.py <work_dir> <stem_variant>
Writes 01_stems/<variant>_dry/: dry versions of the stems in MODELS, the rest hard-linked.
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


def dry_version(wav: Path, model_file: str) -> tuple:
    """Run one dereverb model on a stem; return (dry audio, sr)."""
    with tempfile.TemporaryDirectory() as tmp:
        # ponytail: reloads the model per stem (3 stems, a few s each); cache if stems grow
        sep = Separator(model_file_dir=str(MODEL_DIR), output_dir=tmp, normalization_threshold=1.0)
        sep.load_model(model_file)
        files = [Path(tmp) / Path(f).name for f in sep.separate(str(wav))]
        dry = next(f for f in files if stem_name(f) in DRY_NAMES)
        return sf.read(dry, dtype="float32")


def main(work: Path, variant: str) -> None:
    """Dereverb every stem listed in MODELS; hard-link the others."""
    src, dst = work / "01_stems" / variant, work / "01_stems" / f"{variant}_dry"
    dst.mkdir(parents=True, exist_ok=True)
    for wav in src.glob("*.wav"):
        model_file = MODELS.get(wav.stem)
        if model_file is None:
            if not (dst / wav.name).exists():
                os.link(wav, dst / wav.name)
            continue
        audio, sr = dry_version(wav, model_file)
        sf.write(dst / wav.name, audio, sr, subtype="FLOAT")
        print(f"{wav.stem}: dry via {model_file}", flush=True)
    meta = json.loads((src / "stems.json").read_text())
    (dst / "stems.json").write_text(json.dumps({**meta, "dereverb": MODELS}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
