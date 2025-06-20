import sys
import os
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from helper.load_config import load_config
from helper.setup_mlflow import setup_mlflow
from helper.resolve_config_path import resolve_config_path

# Evaluate model metrics
def evaluate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

# Compare against thresholds
def is_acceptable(actual, expected):
    return all(actual[m] >= expected.get(m, 0) for m in expected)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def log_json_artifact(client, run_id, path):
    if os.path.exists(path):
        client.log_artifact(run_id, local_path=path)
        print(f"📁 Logged artifact: {path}")

def evaluate_all_models():
    config_path = resolve_config_path()
    print(f"Using config path: {config_path}")
    config = load_config(config_path)
    setup_mlflow(config)

    output_dir = Path(config.get("OUTPUT_DIR", config['data']['output_dir']))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test = pd.read_csv(output_dir / "X_test.csv")
    y_test = pd.read_csv(output_dir / "y_test.csv")

    client = mlflow.tracking.MlflowClient()
    for model_cfg in config.get("models", []):
        print(config.get("models"))
        exp = mlflow.get_experiment_by_name(model_cfg["experiment_name"])
        if not exp:
            print(f"⚠️ Experiment not found: {model_cfg['experiment_name']}")
            continue

        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
        if not runs:
            print(f"⚠️ No runs found for {model_cfg['name']}")
            continue

        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        print(model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        y_pred = model.predict(X_test)

        # Compute and evaluate metrics
        actual_metrics = evaluate_metrics(y_test, y_pred)
        expected_metrics = model_cfg.get("metrics_threshold", {})
        accepted = is_acceptable(actual_metrics, expected_metrics)

        # Log metrics and artifacts back to the same run
        print(f"🔁 Logging evaluation into run: {run_id}")
        for key, val in actual_metrics.items():
            client.log_metric(run_id, key, val)

        # Log acceptance decision
        client.set_tag(run_id, "evaluation_status", "accepted" if accepted else "rejected")
        client.set_tag(run_id, "evaluated_model", model_cfg["name"])

        # Log evaluation JSON artifact
        eval_json = {
            "model_name": model_cfg["name"],
            "evaluated_run_id": run_id,
            "accepted": accepted,
            "actual_metrics": actual_metrics,
            "expected_thresholds": config["metrics_threshold"]
        }

        eval_path = output_dir / f"{model_cfg['name']}_evaluation_result.json"
        save_json(eval_path, eval_json)

        thresholds_path = output_dir / f"{model_cfg['name']}_thresholds.json"
        save_json(thresholds_path, expected_metrics)

        # Optional: input/output schema logging (from config)
        if "model_input_schema" in model_cfg:
            input_schema_path = output_dir / f"{model_cfg['name']}_input_schema.json"
            save_json(input_schema_path, model_cfg["model_input_schema"])
            log_json_artifact(client, run_id, input_schema_path)

        if "model_output_schema" in model_cfg:
            output_schema_path = output_dir / f"{model_cfg['name']}_output_schema.json"
            save_json(output_schema_path, model_cfg["model_output_schema"])
            log_json_artifact(client, run_id, output_schema_path)

        # Log all JSON artifacts to original training run
        log_json_artifact(client, run_id, eval_path)
        log_json_artifact(client, run_id, thresholds_path)

        # Log metrics & tags back to training run
        for key, val in actual_metrics.items():
            client.log_metric(run_id, key, val)

        client.set_tag(run_id, "evaluation_status", "accepted" if accepted else "rejected")
        client.set_tag(run_id, "evaluated_model", model_cfg["name"])

        print(f"✅ Model {model_cfg['name']} evaluation logged under run {run_id} — {'ACCEPTED' if accepted else 'REJECTED'}")

if __name__ == "__main__":
    evaluate_all_models()