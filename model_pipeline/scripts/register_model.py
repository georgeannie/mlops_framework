import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import json

from helper.load_config import load_config
from helper.setup_mlflow import setup_mlflow
from helper.resolve_config_path import resolve_config_path
from helper.get_model_config import get_model_config

def register_model_if_accepted(model_key: str, config: dict, client: MlflowClient):
    model_cfg = get_model_config(config, model_key)
    model_name = model_cfg["model_name"]
    experiment_name = model_cfg["experiment_name"]

    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)

    if not runs:
        print(f"⚠️ No runs found for experiment '{experiment_name}'")
        return

    run = runs[0]
    run_id = run.info.run_id
    tags = run.data.tags
    status = tags.get("evaluation_status", "").lower()

    if status != "accepted":
        print(f"🚫 Skipping registration for '{model_name}' – evaluation_status='{status}'")
        return

    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Set additional tags
    client.set_tag(run_id, "registered_model_version", registered_model.version)
    client.set_tag(run_id, "registered_model_name", model_name)
    client.set_tag(run_id, "registered_stage", "None")

    print(f"✅ Registered model '{model_name}' as version {registered_model.version}")

    # Save model info
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = {
        "model_name": model_name,
        "run_id": run_id,
        "version": registered_model.version,
        "status": "registered"
    }

    info_path = output_dir / f"registered_model_info_{model_name}.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    mlflow.log_artifact(str(info_path))

def main():
    config_path = resolve_config_path()
    config = load_config(config_path)
    setup_mlflow(config)
    client = MlflowClient()

    for model_key in ["model_lr", "model_rf"]:
        register_model_if_accepted(model_key, config, client)

if __name__ == "__main__":
    main()
