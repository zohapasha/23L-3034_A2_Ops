import argparse
import json
import shutil

import mlflow
from mlflow import MlflowClient

from config import MODEL_NAME, ROOT, start_mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true")
args = parser.parse_args()
start_mlflow(local=args.local)
client = MlflowClient()

try:
    version = client.get_model_version_by_alias(MODEL_NAME, "champion")
except Exception:
    version = next(v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
                   if v.tags.get("alias") == "champion")

run = client.get_run(version.run_id)
path = mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path="keras_model/best.keras",
                                           dst_path=str(ROOT / "outputs" / "download"))
dest = ROOT / "models" / "champion.keras"
dest.parent.mkdir(exist_ok=True)
shutil.copy(path, dest)

info = {
    "model": MODEL_NAME,
    "version": version.version,
    "run_id": version.run_id,
    "threshold": run.data.metrics.get("threshold", 0.5),
    "img_size": int(run.data.params.get("cfg_img_size", 160)),
}
(ROOT / "models" / "champion.json").write_text(json.dumps(info, indent=2))
print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
print(json.dumps(info, indent=2))
print(f"\nadd to .env:  THRESHOLD={info['threshold']}  IMG_SIZE={info['img_size']}  MODEL_VERSION=v{info['version']}")