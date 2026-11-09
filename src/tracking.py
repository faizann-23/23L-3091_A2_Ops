"""Shared MLflow / DagsHub setup. Secrets come from environment variables (.env), never from code."""
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

DAGSHUB_REPO = "23L-3091_A2_Ops"


def setup_tracking(experiment, target="dagshub"):
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if target == "local":
        mlflow.set_tracking_uri("sqlite:///mlflow_local.db")
    else:
        user, token = os.environ.get("DAGSHUB_USER"), os.environ.get("DAGSHUB_TOKEN")
        if not user or not token:
            raise SystemExit("DAGSHUB_USER / DAGSHUB_TOKEN missing: put them in the .env file.")
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", user)
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)
        mlflow.set_tracking_uri(f"https://dagshub.com/{user}/{DAGSHUB_REPO}.mlflow")
    mlflow.set_experiment(experiment)
    print("Tracking URI :", mlflow.get_tracking_uri())
    print("Experiment   :", experiment)