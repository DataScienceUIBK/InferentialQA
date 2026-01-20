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
SUBDIR = "results/baselines_retriever/msmarco"
DEST = Path.cwd()

DOWNLOAD_DIR = DEST / "_hf_download"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def zsd_to_json_path(zsd_path: Path) -> Path:
    """
    Force output to be a .json file.
    Examples:
      foo.zsd        -> foo.json
      foo.bar.zsd    -> foo.bar.json
      foo.json.zsd   -> foo.json
    """
    stem = zsd_path.name[:-4]  # remove ".zsd"
    if stem.endswith(".json"):
        return zsd_path.with_name(stem)
    return zsd_path.with_name(stem + ".json")


def decompress_zsd_to_json(src_zsd: Path) -> Path:
    out_json = zsd_to_json_path(src_zsd)
    dctx = zstd.ZstdDecompressor()
    with src_zsd.open("rb") as fin, out_json.open("wb") as fout:
        dctx.copy_stream(fin, fout)
    return out_json


def make_dest_name(json_path: Path) -> str:
    """
    New filename = <parentdir>_<originalfilename>
    """
    parent = json_path.parent.name
    return f"{parent}_msmarco_output_100.json"


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

# ---- Decompress all .zsd recursively ----
zsd_files = list(target_dir.rglob("*.zsd"))
if not zsd_files:
    print(f"No .zsd files found under: {target_dir}")

for zsd_file in zsd_files:
    print(f"Decompressing: {zsd_file}")
    out_json = decompress_zsd_to_json(zsd_file)
    zsd_file.unlink()  # delete compressed
    print(f"  -> wrote JSON: {out_json} (deleted {zsd_file.name})")

# ---- Move JSONs: rename using parent folder name ----
json_files = list(target_dir.rglob("*.json"))
if not json_files:
    print(f"No .json files found under: {target_dir}")

for jf in json_files:
    new_name = make_dest_name(jf)
    dest_path = DEST / new_name

    if dest_path.exists():
        raise SystemExit(
            f"Collision even after renaming: {dest_path}\n"
            f"Source: {jf}"
        )

    shutil.move(str(jf), str(dest_path))
    print(f"Moved: {jf} -> {dest_path}")

# ---- Remove download directory completely ----
shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
print(f"\n🧹 Removed directory: {DOWNLOAD_DIR}")

print("\n✅ Done.")
print(f"All JSON files are now in: {DEST}")
