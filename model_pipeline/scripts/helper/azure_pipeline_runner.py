import subprocess
import yaml
from string import Template
import os
from pathlib import Path
import hashlib
import tempfile

AZUREML_COMPONENT_PREFIX = "azureml"

def get_component_hash(yaml_path: str) -> str:
    """Generate a hash based on the content of the component YAML."""
    with open(yaml_path, 'rb') as f:
        content = f.read()
        return hashlib.sha256(content).hexdigest()

def get_latest_component_version_number(component_name: str) -> str:
    """Get the latest version number (just the version, not full ID)."""
    try:
        output = subprocess.check_output([
            "az", "ml", "component", "list",
            "--name", component_name,
            "--query", "[].version",
            "-o", "tsv"
        ], text=True)
        versions = output.strip().splitlines()
        if versions:
            return sorted(versions, key=lambda v: [int(s) if s.isdigit() else s for s in v.split('.')])[-1]
    except subprocess.CalledProcessError:
        return None

def get_registered_component_hash(component_name: str) -> str:
    """Fetch the hash tag of the latest registered component (if any)."""
    version = get_latest_component_version_number(component_name)
    if not version:
        return None
    try:
        output = subprocess.check_output([
            "az", "ml", "component", "show",
            "--name", component_name,
            "--version", version,
            "--query", "tags.hash",
            "-o", "tsv"
        ], text=True).strip()
        return output if output else None
    except subprocess.CalledProcessError:
        return None

def register_component_if_needed(component_yaml_path: str) -> str:
    with open(component_yaml_path) as f:
        component_yaml = yaml.safe_load(f)
        component_name = component_yaml.get("name")

    new_hash = get_component_hash(component_yaml_path)
    existing_hash = get_registered_component_hash(component_name)

    if existing_hash == new_hash:
        version = get_latest_component_version_number(component_name)
        component_id = f"{AZUREML_COMPONENT_PREFIX}:{component_name}:{version}"
        print(f"✅ No changes in {component_name}, using existing: {component_id}")
        return component_id

    print(f"📦 Registering updated component: {component_name}")

    component_dir = Path(component_yaml_path).parent.resolve()

    # Inject base_path into the component YAML
    component_yaml["base_path"] = str(component_dir)

    # Preserve relative paths but include base_path so they resolve properly
    if "code" in component_yaml:
        relative_code_path = Path(component_yaml["code"])
       # component_yaml["code"] = str(relative_code_path)
        component_yaml["code"] = str(Path(component_yaml_path).parent / component_yaml["code"])

    if "environment" in component_yaml and "conda_file" in component_yaml["environment"]:
        relative_conda_path = Path(component_yaml["environment"]["conda_file"])
#        component_yaml["environment"]["conda_file"] = str(relative_conda_path)
        component_yaml["environment"]["conda_file"] = str(Path(component_yaml_path).parent / component_yaml["environment"]["conda_file"])

    # Add or update tags
    component_yaml.setdefault("tags", {})["hash"] = new_hash

    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        yaml.dump(component_yaml, tmp)
        tmp_path = tmp.name

    subprocess.run([
        "az", "ml", "component", "create",
        "--file", tmp_path
    ], check=True)

    version = get_latest_component_version_number(component_name)
    component_id = f"{AZUREML_COMPONENT_PREFIX}:{component_name}:{version}"
    print(f"✅ Registered: {component_id}")
    return component_id

def render_pipeline_yaml(template_path, output_path, substitutions):
    with open(template_path, 'r') as file:
        raw_template = file.read()

    # Escape AzureML Jinja-like placeholders before using string.Template
    safe_template_str = raw_template.replace("${{", "%%AZURE_OPEN%%").replace("}}", "%%AZURE_CLOSE%%")

    # Substitute only the component_id variables (e.g., ${train_component_id})
    template = Template(safe_template_str)
    rendered = template.substitute(substitutions)

    # Restore AzureML placeholders back
    final_yaml = rendered.replace("%%AZURE_OPEN%%", "${{").replace("%%AZURE_CLOSE%%", "}}")

    with open(output_path, 'w') as f:
        f.write(final_yaml)

    print(f"✅ Rendered pipeline YAML to {output_path}")

def run_azure_pipeline(config):
    jobs_dir = Path("jobs")
    template_path = jobs_dir / "pipeline_job_template.yaml"
    output_path = jobs_dir / "pipeline_job.yaml"

    substitutions = {}
    print("🔁 Resolving components dynamically from folder...")

    for file in jobs_dir.glob("*_job.yaml"):
        if file.name in ["pipeline_job_template.yaml", "pipeline_job.yaml"]:
            continue

        component_name = file.stem.replace("_job", "")  # e.g., train_job.yaml → train
        component_id = register_component_if_needed(str(file))
        substitutions[f"{component_name}_component_id"] = component_id
        print(f"🔧 Registered or found: {component_name} → {component_id}")

    if not substitutions:
        raise ValueError("❌ No component YAMLs found to register!")

    render_pipeline_yaml(
        template_path=template_path,
        output_path=output_path,
        substitutions=substitutions
    )

    print("🚀 Launching pipeline job...")
    subprocess.run([
        "az", "ml", "job", "create",
        "--file", str(output_path),
        "--set", f"inputs.config_path={config['job']['config_path']}",
        "--set", f"inputs.compute_target={config['platform']['compute']}"
    ], check=True)
