"""This module is for seting up MLflow logging."""
import mlflow
tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)
