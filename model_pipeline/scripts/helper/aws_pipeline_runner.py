
def run_aws_pipeline(config):
    import boto3
    import argparse
    import yaml
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.steps import TrainingStep
    from sagemaker.inputs import TrainingInput
    from sagemaker.workflow.parameters import ParameterString
    import sagemaker
    
    session = sagemaker.Session()
    role = config["platform"].get("role", "arn:aws:iam::<account-id>:role/<sagemaker-execution-role>")
    region = config["platform"].get("region", "us-east-1")

    # Define parameters
    config_path_param = ParameterString(name="ConfigPath", default_value=config["job"]["config_path"])

    # Define estimator (equivalent to "registering a component")
    sklearn_estimator = SKLearn(
        entry_point="scripts/train_lr.py",
        role=role,
        instance_type="ml.m5.large",
        framework_version="0.23-1",
        base_job_name=config["job"]["base_job_name"],
        source_dir=".",  # Assuming full context includes config/scripts/etc.
        hyperparameters={
            "config": config_path_param
        },
        sagemaker_session=session
    )

    # Dummy input to trigger job (required even if script loads local data)
    train_input = TrainingInput("s3://your-bucket/dummy-data", content_type="text/csv")

    # Define training step
    train_step = TrainingStep(
        name="TrainModel",
        estimator=sklearn_estimator,
        inputs={"train": train_input}
    )

    # Build and execute pipeline
    pipeline = Pipeline(
        name="PlatformAgnosticPipeline",
        parameters=[config_path_param],
        steps=[train_step],
        sagemaker_session=session
    )

    print("✅ Starting SageMaker pipeline execution...")
    pipeline.upsert(role_arn=role)
    execution = pipeline.start(parameters={"ConfigPath": config["job"]["config_path"]})
    execution.wait()