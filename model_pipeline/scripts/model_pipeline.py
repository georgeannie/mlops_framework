# pipeline_job.py
from azure.ai.ml import MLClient, command, dsl, Input, Output
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<SUBSCRIPTION_ID>",
    resource_group_name="<RESOURCE_GROUP>",
    workspace_name="<WORKSPACE_NAME>"
)

# Define train jobs (outputs: run_id, test files)
train_rf_job = command(
    code=".",
    command="python train_rf.py",
    environment="azureml:AzureML-sklearn-1.1-ubuntu20.04-py38-cpu:1",
    outputs={
        "rf_run_id": Output(type="uri_file", path="rf_run_id.txt"),
        "X_test_rf": Output(type="uri_file", path="X_test_rf.csv"),
        "y_test_rf": Output(type="uri_file", path="y_test_rf.csv"),
    },
    compute="cpu-cluster"
)

train_lr_job = command(
    code=".",
    command="python train_lr.py",
    environment="azureml:AzureML-sklearn-1.1-ubuntu20.04-py38-cpu:1",
    outputs={
        "lr_run_id": Output(type="uri_file", path="lr_run_id.txt"),
        "X_test_lr": Output(type="uri_file", path="X_test_lr.csv"),
        "y_test_lr": Output(type="uri_file", path="y_test_lr.csv"),
    },
    compute="cpu-cluster"
)

# Validation jobs (one per model)
validate_rf_job = command(
    code=".",
    command="python validate_model.py ${inputs.run_id} ${inputs.X_test} ${inputs.y_test} rf_register.flag rf_metrics.json",
    environment="azureml:AzureML-sklearn-1.1-ubuntu20.04-py38-cpu:1",
    inputs={
        "run_id": train_rf_job.outputs["rf_run_id"],
        "X_test": train_rf_job.outputs["X_test_rf"],
        "y_test": train_rf_job.outputs["y_test_rf"],
    },
    outputs={
        "rf_flag": Output(type="uri_file", path="rf_register.flag"),
        "rf_metrics": Output(type="uri_file", path="rf_metrics.json"),
    },
    compute="cpu-cluster"
)

validate_lr_job = command(
    code=".",
    command="python validate_model.py ${inputs.run_id} ${inputs.X_test} ${inputs.y_test} lr_register.flag lr_metrics.json",
    environment="azureml:AzureML-sklearn-1.1-ubuntu20.04-py38-cpu:1",
    inputs={
        "run_id": train_lr_job.outputs["lr_run_id"],
        "X_test": train_lr_job.outputs["X_test_lr"],
        "y_test": train_lr_job.outputs["y_test_lr"],
    },
    outputs={
        "lr_flag": Output(type="uri_file", path="lr_register.flag"),
        "lr_metrics": Output(type="uri_file", path="lr_metrics.json"),
    },
    compute="cpu-cluster"
)

# Register jobs (only proceeds if flag file is present)
register_rf_job = command(
    code=".",
    command="python register_model.py ${inputs.run_id} ${inputs.flag_file} RandomForestModel",
    environment="azureml:AzureML-sklearn-1.1-ubuntu20.04-py38-cpu:1",
    inputs={
        "run_id": train_rf_job.outputs["rf_run_id"],
        "flag_file": validate_rf_job.outputs["rf_flag"]
    },
    compute="cpu-cluster"
)

register_lr_job = command(
    code=".",
    command="python register_model.py ${inputs.run_id} ${inputs.flag_file} LogisticRegressionModel",
    environment="azureml:AzureML-sklearn-1.1-ubuntu20.04-py38-cpu:1",
    inputs={
        "run_id": train_lr_job.outputs["lr_run_id"],
        "flag_file": validate_lr_job.outputs["lr_flag"]
    },
    compute="cpu-cluster"
)

# Orchestrate the pipeline
@dsl.pipeline(
    default_compute="cpu-cluster"
)
def parallel_train_validate_register_pipeline():
    rf_train = train_rf_job()
    lr_train = train_lr_job()
    rf_validate = validate_rf_job(run_id=rf_train.outputs["rf_run_id"], X_test=rf_train.outputs["X_test_rf"], y_test=rf_train.outputs["y_test_rf"])
    lr_validate = validate_lr_job(run_id=lr_train.outputs["lr_run_id"], X_test=lr_train.outputs["X_test_lr"], y_test=lr_train.outputs["y_test_lr"])
    register_rf_job(run_id=rf_train.outputs["rf_run_id"], flag_file=rf_validate.outputs["rf_flag"])
    register_lr_job(run_id=lr_train.outputs["lr_run_id"], flag_file=lr_validate.outputs["lr_flag"])

pipeline = parallel_train_validate_register_pipeline()
ml_client.jobs.create_or_update(pipeline)
