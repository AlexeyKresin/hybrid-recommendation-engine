from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_DIR = RAW_DIR / "movielens"
ZIP_PATH = RAW_DIR / "ml-100k.zip"
EXTRACT_DIR = RAW_DIR / "ml-100k"

REQUIRED_FILES = ["u.data", "u.item", "u.user"]


def dataset_ready() -> bool:
    return all((TARGET_DIR / file_name).exists() for file_name in REQUIRED_FILES)


def download_zip() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from:\n{DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    print(f"Downloaded to: {ZIP_PATH}")


def extract_zip() -> None:
    print(f"Extracting: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)
    print(f"Extracted to: {EXTRACT_DIR}")


def move_required_files() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for file_name in REQUIRED_FILES:
        source = EXTRACT_DIR / file_name
        destination = TARGET_DIR / file_name

        if not source.exists():
            raise FileNotFoundError(f"Expected file not found after extraction: {source}")

        shutil.copy2(source, destination)
        print(f"Copied {file_name} -> {destination}")


def cleanup(remove_zip: bool = True, remove_extracted_folder: bool = True) -> None:
    if remove_zip and ZIP_PATH.exists():
        ZIP_PATH.unlink()
        print(f"Removed zip file: {ZIP_PATH}")

    if remove_extracted_folder and EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
        print(f"Removed extracted folder: {EXTRACT_DIR}")


def main() -> None:
    print("Setting up MovieLens 100K dataset...")

    if dataset_ready():
        print("Dataset already exists. Nothing to do.")
        return

    try:
        download_zip()
        extract_zip()
        move_required_files()
        cleanup()
        print("\nDataset setup completed successfully.")
        print(f"Files are available in: {TARGET_DIR}")
    except Exception as exc:
        print(f"\nDataset setup failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()