#specify platform method to connect to mlflow - AWS will be different

def setup_mlflow(config, experiment_name=None):
    if config["platform"].get("use_azureml", False):
        from azureml.core import Workspace
        ws = Workspace.from_config()
        tracking_uri = ws.get_mlflow_tracking_uri()
    else:
        tracking_uri = config["mlflow"]["tracking_uri"]

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    
