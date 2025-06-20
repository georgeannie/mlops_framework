import mlflow, pandas as pd
from pathlib import Path
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helper.load_config import load_config
from helper.setup_mlflow import setup_mlflow
from helper.resolve_config_path import resolve_config_path
from helper.get_model_config import get_model_config

# Load config
config_path = resolve_config_path()
print(f"Using config path: {config_path}")
config = load_config(config_path)
setup_mlflow(config)
model_rf = get_model_config(config, "model_rf")
mlflow.set_experiment(model_rf["experiment_name"])

mlflow.set_experiment(model_rf["experiment_name"])

output_dir = Path(config.get("OUTPUT_DIR", config['data']['output_dir']))
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------
# Load Data
# ------------------------
data_path = config["data"]["path"]
target_col = config["data"]["target_column"]
output_dir = Path(config["data"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Train and Log Model
# ------------------------
hparams = model_rf["hyperparameters"]

with mlflow.start_run(run_name=model_rf["run_name"]) as run:
    clf = RandomForestClassifier(
        n_estimators=hparams.get("n_estimators", 100),
        max_depth=hparams.get("max_depth", None),
        random_state=hparams.get("random_state", 42)
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model"
    )

    # Log hyperparameters
    mlflow.log_params({
        "n_estimators": clf.n_estimators,
        "max_depth": clf.max_depth,
        "random_state": clf.random_state
    })

    # Log metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }
    mlflow.log_metrics(metrics)

    # Save and log test artifacts
    X_test_path = output_dir / "X_test_rf.csv"
    y_test_path = output_dir / "y_test_rf.csv"
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    mlflow.log_artifact(str(X_test_path))
    mlflow.log_artifact(str(y_test_path))

    # Save run ID
    run_id_path = output_dir / "run_id_rf.txt"
    with open(run_id_path, "w") as f:
        f.write(run.info.run_id)
    mlflow.log_artifact(str(run_id_path))

    # Save and log schemas
    input_schema_path = output_dir / "model_input_schema_rf.json"
    output_schema_path = output_dir / "model_output_schema_rf.json"

    with open(input_schema_path, "w") as f:
        json.dump(hparams.get("model_input_schema", {}), f, indent=2)
    with open(output_schema_path, "w") as f:
        json.dump(hparams.get("model_output_schema", {}), f, indent=2)

    mlflow.log_artifact(str(input_schema_path))
    mlflow.log_artifact(str(output_schema_path))

    # Apply model tags
    mlflow.set_tags(model_rf.get("model_tags", {}))