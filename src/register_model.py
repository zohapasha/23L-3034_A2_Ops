import argparse

from mlflow import MlflowClient

from config import MODEL_NAME, PROJECT, start_mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true")
args = parser.parse_args()
mlflow = start_mlflow(local=args.local)
client = MlflowClient()

runs = mlflow.search_runs(experiment_names=[PROJECT], order_by=["metrics.val_auc DESC"])
runs = runs[runs["metrics.test_f1"].notna()]
best, second = runs.run_id.iloc[0], runs.run_id.iloc[1]

# second best becomes version 1, best becomes version 2
versions = []
for run_id in [second, best]:
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    versions.append(mv.version)
    print(f"registered {MODEL_NAME} v{mv.version} from run {run_id}")

try:
    client.set_registered_model_alias(MODEL_NAME, "champion", versions[-1])
    print("alias champion ->", versions[-1])
except Exception as e:
    print("aliases not supported, using a tag instead:", e)
    client.set_model_version_tag(MODEL_NAME, versions[-1], "alias", "champion")