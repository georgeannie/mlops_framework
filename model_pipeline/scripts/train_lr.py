import mlflow, pandas as pd
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helper.load_config import load_config
from helper.setup_mlflow import setup_mlflow
from helper.resolve_config_path import resolve_config_path
from helper.get_model_config import get_model_config

# Load config
experiment_name = "Logistic_Regression_Training"

config_path = resolve_config_path()
config = load_config(config_path)
setup_mlflow(config)
model_lr = get_model_config(config, "model_lr")
mlflow.set_experiment(model_lr["experiment_name"])

data_path = config["data"]["path"]
target_col = config["data"]["target_column"]
output_dir = Path(config["data"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hparams = model_lr["hyperparameters"]

with mlflow.start_run(run_name=model_lr["run_name"]) as run:
    lr = LogisticRegression(
        C=hparams.get("C", 1.0),
        solver=hparams.get("solver", "lbfgs"),
        max_iter=hparams.get("max_iter", 100)
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="model"
    )

    # Log hyperparameters
    mlflow.log_params({
        "C": lr.C,
        "solver": lr.solver,
        "max_iter": lr.max_iter
    })

    # Calculate and log metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }
    mlflow.log_metrics(metrics)

    # Save test set
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    mlflow.log_artifact(str(output_dir / "X_test.csv"))
    mlflow.log_artifact(str(output_dir / "y_test.csv"))

    # Save and log run ID
    run_id_path = output_dir / "run_id.txt"
    with open(run_id_path, "w") as f:
        f.write(run.info.run_id)
    mlflow.log_artifact(str(run_id_path))

    # Save and log model schemas
    model_input_schema = hparams.get("model_input_schema", {})
    model_output_schema = hparams.get("model_output_schema", {})

    input_schema_path = output_dir / "model_input_schema_lr.json"
    output_schema_path = output_dir / "model_output_schema_lr.json"
    
    with open(input_schema_path, "w") as f:
        json.dump(model_input_schema, f, indent=2)
    with open(output_schema_path, "w") as f:
        json.dump(model_output_schema, f, indent=2)

    mlflow.log_artifact(str(input_schema_path))
    mlflow.log_artifact(str(output_schema_path))

    # Log model tags
    mlflow.set_tags(model_lr.get("model_tags", {}))