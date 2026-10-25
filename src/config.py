import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "splits"
OUTPUT_DIR = ROOT / "outputs"
CLASSES = ["real", "ai"]  # 0 = real, 1 = ai

PROJECT = os.getenv("PROJECT_NAME", "ROLL_A2_Ops")
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", PROJECT + "_model")


def set_seed(seed=42):
    import keras
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)


def start_mlflow(local=False, experiment=None):
    import mlflow
    if local:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    else:
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if not uri:
            raise SystemExit("MLFLOW_TRACKING_URI not set, check your .env file")
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or PROJECT)
    return mlflow


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def data_version():
    # md5 of the dataset from the dvc pointer file
    dvc_file = ROOT / "data" / "raw.dvc"
    if not dvc_file.exists():
        return "untracked"
    for line in dvc_file.read_text().splitlines():
        line = line.strip().lstrip("- ")
        if line.startswith("md5:"):
            return line.split(":", 1)[1].strip()
    return "unknown"
