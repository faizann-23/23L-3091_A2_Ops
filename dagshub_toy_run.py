import os
import sys
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "faizann-23")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "23L-3091_A2_Ops")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
username = os.getenv("MLFLOW_TRACKING_USERNAME", "faizann-23")
token = os.getenv("DAGSHUB_USER_TOKEN")

# Set standard MLflow authentication environment variables directly
os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

import mlflow

print("="*80)
print("SUB-TASK 3.2: DAGSHUB REMOTE TOY EXPERIMENT")
print("="*80)
print(f"Tracking URI : {tracking_uri}")
print(f"Username     : {username}")
print(f"Auth Status  : Token configured via environment variables (.env)")

mlflow.set_tracking_uri(tracking_uri)

experiment_name = "Dagshub_Toy_Run_23L_3091"
mlflow.set_experiment(experiment_name)

print("\n[*] Logging remote run to DagsHub...")
with mlflow.start_run(run_name="dagshub_connection_test"):
    mlflow.log_param("student_roll_no", "23L-3091")
    mlflow.log_param("test_param", "dagshub_remote_verified")
    mlflow.log_metric("remote_connection_status", 1.0)
    mlflow.set_tag("verified_by", "VisionGuard MLOps Pipeline")
    mlflow.set_tag("student", "23L-3091")
    print("[+] Successfully logged toy run directly to DagsHub MLflow remote server!")

print("="*80)
print(f"Open your DagsHub repository in browser:")
print(f"https://dagshub.com/{repo_owner}/{repo_name}/experiments")
print("="*80)
