import json
from pathlib import Path


def open_json(
    rel_path: str,
    *,
    local_root: str | Path = "../__hf_cache",
    base_url: str = "https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/results/",
    force_download: bool = False,
    timeout: int = 60,
):
    """
    Supports .json and .jsonl.

    Given:
        rel_path="baselines/baseline.json"
    Downloads (if needed):
        ".../results/baselines/baseline.zst"
    Decompresses to:
        local_root/baselines/baseline.json

    For jsonl:
        rel_path="x/y.jsonl"
        downloads ".../results/x/y.zst"
        returns f.readlines()
    """
    import requests
    from tqdm import tqdm

    try:
        import zstandard as zstd  # pip install zstandard
    except ImportError as e:
        raise ImportError(
            "Missing dependency: zstandard. Install with: pip install zstandard"
        ) from e

    rel_path = Path(rel_path)
    suffix = rel_path.suffix.lower()
    if suffix not in [".json", ".jsonl"]:
        raise ValueError(f"open_json supports only .json or .jsonl, got: {rel_path}")

    local_root = Path(local_root)
    local_file_path = (local_root / rel_path).resolve()

    # IMPORTANT: compressed path is same without .json/.jsonl
    # baseline.json  -> baseline.zst
    compressed_rel_path = rel_path.with_suffix(".zsd")  # baselines/baseline.zst
    local_zst_path = (local_root / compressed_rel_path).resolve()
    remote_url = base_url.rstrip("/") + "/" + compressed_rel_path.as_posix()

    # If already decompressed locally, load directly
    if local_file_path.exists() and not force_download:
        if suffix == ".json":
            with local_file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        else:  # .jsonl
            with local_file_path.open("r", encoding="utf-8") as f:
                return f.readlines()

    # Ensure directories exist
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress
    if force_download or (not local_zst_path.exists()):
        local_zst_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(remote_url, stream=True, timeout=timeout) as r:
            r.raise_for_status()

            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 1024 * 1024  # 1MB

            with tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {compressed_rel_path.as_posix()}",
            ) as pbar:
                with local_zst_path.open("wb") as out:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            out.write(chunk)
                            pbar.update(len(chunk))

    # Decompress .zst -> target file (.json or .jsonl)
    dctx = zstd.ZstdDecompressor()
    with local_zst_path.open("rb") as compressed, local_file_path.open("wb") as decompressed:
        dctx.copy_stream(compressed, decompressed)

    # Load and return
    if suffix == ".json":
        with local_file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:  # .jsonl
        with local_file_path.open("r", encoding="utf-8") as f:
            return f.readlines()
