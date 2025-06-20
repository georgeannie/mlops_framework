import sys
import os
import mlflow

# Usage: python register_model.py <run_id> <flag_file> <model_name>
def main():
    run_id = sys.argv[1]
    flag_file = sys.argv[2]
    model_name = sys.argv[3]

    if not os.path.exists(flag_file):
        print(f"Registration skipped: Validation flag missing for {model_name}.")
        return

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    print(f"Model {model_name} registered as version {result.version}")

if __name__ == "__main__":
    main()
