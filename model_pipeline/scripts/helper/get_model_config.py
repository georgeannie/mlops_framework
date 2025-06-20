def get_model_config(config, model_name="model_lr"):
    for model in config["models"]:
        if model["name"] == model_name:
            return model
    raise ValueError(f"Model config with name '{model_name}' not found.")