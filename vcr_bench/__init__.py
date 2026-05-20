# Ensure a writable temporary directory exists before torch is imported.
# On some lab machines /tmp is unavailable; tempfile raises FileNotFoundError
# deep inside the torch import chain if none of the standard candidates work.
import os as _os
import pathlib as _pathlib

def _ensure_tmp() -> None:
    if any(_os.environ.get(_k) for _k in ("TMPDIR", "TEMP", "TMP")):
        return
    fallback = _pathlib.Path.home() / ".cache" / "tmp"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        _os.environ["TMPDIR"] = str(fallback)
        _os.environ["TEMP"] = str(fallback)
        _os.environ["TMP"] = str(fallback)
    except OSError:
        pass  # nothing we can do; torch will give its own error

_ensure_tmp()
del _os, _pathlib, _ensure_tmp

from .types import PredictionBundle, VideoSampleRef

__all__ = ["PredictionBundle", "VideoSampleRef"]
