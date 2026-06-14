from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data", help="Output directory (will create PlantVillage/ under it)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle API requires ~/.kaggle/kaggle.json
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "kaggle package not available. Run: pip install kaggle\n"
            f"Original error: {e}"
        )

    api = KaggleApi()
    api.authenticate()

    dataset = "arjuntejaswi/plant-village"
    api.dataset_download_files(dataset, path=str(out_dir), quiet=False)

    zip_path = out_dir / "plant-village.zip"
    if not zip_path.exists():
        # Kaggle sometimes names it after the dataset slug.
        candidates = list(out_dir.glob("*.zip"))
        if len(candidates) == 1:
            zip_path = candidates[0]
        else:
            raise FileNotFoundError(f"Could not find downloaded zip in {out_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # If it extracted nested, normalize to out_dir/PlantVillage
    if (out_dir / "PlantVillage").exists():
        print(f"Extracted dataset to {out_dir / 'PlantVillage'}")
        return

    extracted = list(out_dir.glob("**/PlantVillage"))
    if extracted:
        src = extracted[0]
        dst = out_dir / "PlantVillage"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"Extracted dataset to {dst}")
        return

    raise FileNotFoundError("After unzip, could not find a PlantVillage/ directory.")


if __name__ == "__main__":
    main()

