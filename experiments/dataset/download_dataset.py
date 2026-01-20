import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    import zstandard as zstd
except ImportError:
    raise SystemExit(
        "Missing dependency: zstandard\n"
        "Install with: pip install zstandard huggingface_hub"
    )

# ---- Config ----
REPO_ID = "JamshidJDMY/InferentialQA"
SUBDIR = "results/dataset"
DEST = Path.cwd()

DOWNLOAD_DIR = DEST / "_hf_download_dataset"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def compressed_to_json_path(src: Path) -> Path:
    """
    Force output to .json.
    Examples:
      foo.zsd       -> foo.json
      foo.zst       -> foo.json
      foo.bar.zsd   -> foo.bar.json
      foo.json.zsd  -> foo.json
    """
    name = src.name
    if name.endswith(".zsd"):
        base = name[:-4]
    elif name.endswith(".zst"):
        base = name[:-4]
    else:
        base = src.stem

    if base.endswith(".json"):
        return src.with_name(base)
    return src.with_name(base + ".json")


def decompress_to_json(src: Path) -> Path:
    out_json = compressed_to_json_path(src)
    dctx = zstd.ZstdDecompressor()
    with src.open("rb") as fin, out_json.open("wb") as fout:
        dctx.copy_stream(fin, fout)
    return out_json


# ---- Download ----
local_path = Path(
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DOWNLOAD_DIR),
        local_dir_use_symlinks=False,
        allow_patterns=[f"{SUBDIR}/**"],
    )
)

target_dir = local_path / SUBDIR
if not target_dir.exists():
    raise SystemExit(f"Downloaded snapshot does not contain expected folder: {target_dir}")

# ---- Decompress all .zsd/.zst recursively into .json ----
compressed_files = list(target_dir.rglob("*.zsd")) + list(target_dir.rglob("*.zst"))
for comp in compressed_files:
    print(f"Decompressing: {comp}")
    out = decompress_to_json(comp)
    comp.unlink()
    print(f"  -> wrote JSON: {out} (deleted {comp.name})")

# ---- Move all .json files to current directory ----
json_files = list(target_dir.rglob("*.json"))
if not json_files:
    print(f"No .json files found under: {target_dir}")

for src in json_files:
    dest_path = DEST / src.name
    if dest_path.exists():
        raise SystemExit(
            f"Collision: {dest_path} already exists.\n"
            f"Source: {src}\n"
            "Delete/rename the existing file, or modify the script to overwrite/auto-rename."
        )
    shutil.move(str(src), str(dest_path))
    print(f"Moved: {src} -> {dest_path}")

# ---- Cleanup ----
shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
print(f"\n🧹 Removed directory: {DOWNLOAD_DIR}")
print("\n✅ Done.")
