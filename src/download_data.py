"""
Dataset Downloader for SMS Spam Collection (5,574 messages).
Downloads and prepares the dataset directly into data/spam.csv.
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "spam.csv"

# Sources: UCI archive endpoint and github raw fallback
URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
]


def download_dataset():
    """Download and process SMS Spam Collection into data/spam.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_zip = DATA_DIR / "temp_sms_dataset.zip"

    print("[INFO] Attempting to download full SMS Spam Collection dataset...")

    success = False
    for url in URLS:
        try:
            print(f"[INFO] Fetching from: {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )

            if url.endswith(".tsv"):
                # Direct TSV file
                with urllib.request.urlopen(req, timeout=30) as response:
                    df = pd.read_csv(response, sep="\t", header=None, names=["label", "message"])
                    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
                success = True
                break
            else:
                # Zip file
                with urllib.request.urlopen(req, timeout=30) as response, open(temp_zip, "wb") as out_file:
                    out_file.write(response.read())

                with zipfile.ZipFile(temp_zip, "r") as z:
                    for filename in z.namelist():
                        if "SMSSpamCollection" in filename:
                            with z.open(filename) as f:
                                df = pd.read_csv(f, sep="\t", header=None, names=["label", "message"])
                                df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
                                success = True
                                break

                if temp_zip.exists():
                    temp_zip.unlink()

                if success:
                    break

        except Exception as e:
            print(f"[WARNING] Failed downloading from {url}: {e}")
            if temp_zip.exists():
                temp_zip.unlink()

    if success and OUTPUT_CSV.exists():
        df_check = pd.read_csv(OUTPUT_CSV)
        print("\n" + "=" * 55)
        print(" DATASET READY")
        print("=" * 55)
        print(f"File Saved At  : {OUTPUT_CSV}")
        print(f"Total Messages : {len(df_check):,}")
        print(f"Ham Count      : {(df_check['label'] == 'ham').sum():,}")
        print(f"Spam Count     : {(df_check['label'] == 'spam').sum():,}")
        print("=" * 55)
        print("\nNow run: python src/train.py to train on this full dataset!\n")
    else:
        print("[ERROR] Could not automatically download dataset from the web.")
        print("Please download manually from Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
        print(f"And place it as: {OUTPUT_CSV}")


if __name__ == "__main__":
    download_dataset()
