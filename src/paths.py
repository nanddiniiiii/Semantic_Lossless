from pathlib import Path
import os


def project_root() -> Path:
    root = Path(os.environ.get("SEMANTIC_LOSSLESS_ROOT", Path(__file__).resolve().parents[1])).resolve()
    if not (root / "sources.lock.json").is_file():
        raise FileNotFoundError("Run from an editable repository checkout or set SEMANTIC_LOSSLESS_ROOT")
    return root


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()
