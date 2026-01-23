"""
Unified experiment tracking with MLflow.

Provides utilities for:
- Creating MLflow run context
- Logging hyperparameters from configs
- Logging model artifacts
- Logging metrics and confusion matrices
- Logging training plots and model summaries
"""

import mlflow
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from textgnn.config_class import Config
from textgnn.logger import setup_logger

logger = setup_logger(__name__)


def _safe_log_metric(name: str, value: Any, step: Optional[int] = None):
    """Safely log a metric to MLflow, skipping non-finite values."""
    if value is None or not np.isfinite(value):
        return
    mlflow.log_metric(name, float(value), step=step)


class ExperimentTracker:
    """
    Centralized experiment tracking using MLflow.

    Usage:
        with ExperimentTracker(config) as tracker:
            # Training happens
            tracker.log_metrics({"train_loss": 0.5}, step=1)
            # After training
            tracker.log_model(model)
            tracker.log_confusion_matrix(y_true, y_pred, labels)
    """

    def __init__(self, config: Config):
        """
        Initialize experiment tracker.

        Args:
            config: Pydantic Config object with experiment, dataset, and model configs
        """
        self.config = config
        self.experiment_name = config.experiment_name
        self.run_name = config.run_name
        self.run = None

    def __enter__(self):
        """Start MLflow run context."""
        # Set experiment (creates if doesn't exist)
        mlflow.set_experiment(self.experiment_name)

        # Start run with system metrics logging enabled
        self.run = mlflow.start_run(run_name=self.run_name, log_system_metrics=True)
        logger.info(f"Started MLflow run: {self.run.info.run_id}")
        logger.info("System metrics logging enabled (CPU, RAM, GPU, VRAM)")

        # Log all hyperparameters
        self._log_config_params()

        # Log config as JSON artifact
        self._log_config_json()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run context."""
        if self.run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run.info.run_id}")

    def _log_config_params(self):
        """Log all config parameters as MLflow params."""
        try:
            # Model params
            mlflow.log_param("model_type", self.config.model_conf.model_type)
            for key, value in self.config.model_conf.common_params.items():
                mlflow.log_param(f"common_{key}", value)
            for key, value in self.config.model_conf.model_specific_params.items():
                mlflow.log_param(f"model_{key}", value)

            # Dataset params
            mlflow.log_param("dataset_name", self.config.dataset.name)
            mlflow.log_param("train_split", self.config.dataset.tvt_split[0])
            mlflow.log_param("val_split", self.config.dataset.tvt_split[1])
            mlflow.log_param("test_split", self.config.dataset.tvt_split[2])
            mlflow.log_param("random_seed", self.config.dataset.random_seed)

            # Preprocessing params
            preprocess = self.config.dataset.preprocess
            mlflow.log_param("remove_stopwords", preprocess.remove_stopwords)
            mlflow.log_param("remove_rare_words", preprocess.remove_rare_words)

            # Encoding params (if available)
            if self.config.dataset.rnn_encoding:
                encoding = self.config.dataset.rnn_encoding
                mlflow.log_param("embedding_dim", encoding.embedding_dim)
                mlflow.log_param("tokens_trained_on", encoding.tokens_trained_on)
            elif self.config.dataset.gnn_encoding:
                encoding = self.config.dataset.gnn_encoding
                mlflow.log_param("x_type", encoding.x_type)
                mlflow.log_param("window_size", encoding.window_size)

            # Tags for easy filtering
            mlflow.set_tag("model_type", self.config.model_conf.model_type)
            mlflow.set_tag("dataset", self.config.dataset.name)

            logger.info("Logged hyperparameters to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log config params: {e}")

    def _log_config_json(self):
        """Log full config as JSON artifact."""
        try:
            config_dict = {
                "experiment_name": self.config.experiment_name,
                "run_name": self.config.run_name,
                "dataset": self.config.dataset.model_dump(),
                "model_config": self.config.model_conf.model_dump()
            }

            config_path = "config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            mlflow.log_artifact(config_path)
            Path(config_path).unlink()

            logger.info("Logged config JSON to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log config JSON: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow safely.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (e.g., epoch number)
        """
        try:
            for key, value in metrics.items():
                _safe_log_metric(key, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a single metric to MLflow safely.
        """
        try:
            _safe_log_metric(name, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metric {name}: {e}")

    def log_model(self, model: torch.nn.Module, model_name: str = "best_model"):
        """
        Save model checkpoint as MLflow artifact.

        Args:
            model: PyTorch model
            model_name: Name for the saved model file (without extension)
        """
        try:
            # Save model state dict
            model_path = f"{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)

            # Clean up local file
            Path(model_path).unlink()

            logger.info(f"Logged model checkpoint: {model_name}.pt")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")

    def log_model_summary(self, model: torch.nn.Module):
        """
        Log model architecture summary as text artifact.

        Args:
            model: PyTorch model
        """
        try:
            summary_path = "model_architecture.txt"

            with open(summary_path, "w") as f:
                f.write("Model Architecture\n")
                f.write("=" * 80 + "\n\n")
                f.write(str(model) + "\n\n")

                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                f.write(f"Total parameters: {total_params:,}\n")
                f.write(f"Trainable parameters: {trainable_params:,}\n")
                f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")

            mlflow.log_artifact(summary_path)
            Path(summary_path).unlink()

            logger.info("Logged model architecture summary")
        except Exception as e:
            logger.warning(f"Failed to log model summary: {e}")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        name: str = "confusion_matrix"
    ):
        """
        Generate and log confusion matrix as artifact.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            labels: Class labels for axes (optional)
            name: Name for the plot file
        """
        try:
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels if labels else range(len(cm)),
                yticklabels=labels if labels else range(len(cm))
            )
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title(f'Confusion Matrix - {name.replace("_", " ").title()}')
            plt.tight_layout()

            # Save and log
            plot_path = f"{name}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(plot_path)
            plt.close()

            # Clean up
            Path(plot_path).unlink()

            logger.info(f"Logged confusion matrix: {name}.png")
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def log_training_curves(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Optional[Dict[str, List[float]]] = None
    ):
        """
        Generate and log training/validation curves.

        Args:
            train_metrics: Dict of metric_name -> list of values (one per epoch)
            val_metrics: Optional dict of validation metric_name -> list of values
        """
        try:
            # Plot loss curve
            if "loss" in train_metrics:
                plt.figure(figsize=(10, 6))
                epochs = range(1, len(train_metrics["loss"]) + 1)

                plt.plot(epochs, train_metrics["loss"], 'b-', label='Train Loss', linewidth=2)
                if val_metrics and "loss" in val_metrics:
                    plt.plot(epochs, val_metrics["loss"], 'r-', label='Val Loss', linewidth=2)

                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                loss_path = "loss_curve.png"
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(loss_path)
                plt.close()
                Path(loss_path).unlink()

            # Plot accuracy curve
            if "accuracy" in train_metrics:
                plt.figure(figsize=(10, 6))
                epochs = range(1, len(train_metrics["accuracy"]) + 1)

                plt.plot(epochs, train_metrics["accuracy"], 'b-', label='Train Accuracy', linewidth=2)
                if val_metrics and "accuracy" in val_metrics:
                    plt.plot(epochs, val_metrics["accuracy"], 'r-', label='Val Accuracy', linewidth=2)

                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Training and Validation Accuracy')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                acc_path = "accuracy_curve.png"
                plt.savefig(acc_path, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(acc_path)
                plt.close()
                Path(acc_path).unlink()

            logger.info("Logged training curves")
        except Exception as e:
            logger.warning(f"Failed to log training curves: {e}")

    def log_roc_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        labels: Optional[List[str]] = None,
        name: str = "roc_curve"
    ):
        """
        Generate and log ROC curve. Handles both binary and multi-class cases.
        """
        try:
            if np.unique(y_true).size < 2:
                logger.warning("Skipping ROC curve generation because y_true contains only one class.")
                return

            plt.figure(figsize=(10, 8))
            
            # Binary classification case
            if y_probs.ndim == 1 or y_probs.shape[1] <= 2:
                y_score = y_probs if y_probs.ndim == 1 else y_probs[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
                self.log_metric(f"{name}_auc", roc_auc)

            # Multi-class classification case
            else:
                n_classes = y_probs.shape[1]
                class_labels = labels if labels else [f"Class {i}" for i in range(n_classes)]
                y_true_bin = label_binarize(y_true, classes=range(n_classes))

                fpr, tpr, roc_auc = {}, {}, {}
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-avg (AUC = {roc_auc["macro"]:.3f})', color='navy', linestyle='--', linewidth=2)
                plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})', color='deeppink', linestyle=':', linewidth=2)

                self.log_metric("roc_auc_macro", roc_auc["macro"])
                self.log_metric("roc_auc_micro", roc_auc["micro"])

            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name.replace("_", " ").title()}')
            plt.legend(loc='lower right', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = f"{name}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(plot_path)
            plt.close()
            Path(plot_path).unlink()
            logger.info(f"Logged ROC curve: {name}.png")
            
        except Exception as e:
            logger.warning(f"Failed to log ROC curve: {e}")

    def log_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        labels: Optional[List[str]] = None,
        name: str = "pr_curve"
    ):
        """
        Generate and log Precision-Recall curve. Handles both binary and multi-class cases.
        """
        try:
            if np.unique(y_true).size < 2:
                logger.warning("Skipping PR curve generation because y_true contains only one class.")
                return

            plt.figure(figsize=(10, 8))

            # Binary classification case
            if y_probs.ndim == 1 or y_probs.shape[1] <= 2:
                y_score = y_probs if y_probs.ndim == 1 else y_probs[:, 1]
                precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
                ap = average_precision_score(y_true, y_score)
                plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.3f})')
                self.log_metric(f"{name}_ap", ap)

            # Multi-class classification case
            else:
                n_classes = y_probs.shape[1]
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                
                precision, recall, avg_precision = {}, {}, {}
                for i in range(n_classes):
                    precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
                    avg_precision[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])

                precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_probs.ravel())
                avg_precision["micro"] = average_precision_score(y_true_bin, y_probs, average="micro")
                
                plt.plot(recall["micro"], precision["micro"], label=f'Micro-avg (AP = {avg_precision["micro"]:.3f})', color='deeppink', linestyle=':', linewidth=2)
                self.log_metric("avg_precision_micro", avg_precision["micro"])

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {name.replace("_", " ").title()}')
            plt.legend(loc='lower left', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = f"{name}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(plot_path)
            plt.close()
            Path(plot_path).unlink()

            logger.info(f"Logged PR curve: {name}.png")
        except Exception as e:
            logger.warning(f"Failed to log PR curve: {e}")

    def log_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        name: str = "per_class_metrics"
    ):
        """
        Generate and log bar chart of precision/recall/F1 per class.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            labels: Class labels for x-axis (optional)
            name: Name for the plot file
        """
        try:
            # Generate classification report as dict
            report = classification_report(y_true, y_pred, output_dict=True)

            # Extract per-class metrics
            classes = sorted([k for k in report.keys() if k.isdigit() or (isinstance(k, str) and k not in ['accuracy', 'macro avg', 'weighted avg'])])
            n_classes = len(classes)
            class_labels = labels if labels else [f"Class {c}" for c in classes]

            precision_vals = [report[c]['precision'] for c in classes]
            recall_vals = [report[c]['recall'] for c in classes]
            f1_vals = [report[c]['f1-score'] for c in classes]

            # Create grouped bar chart
            x = np.arange(n_classes)
            width = 0.25

            fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.8), 6))
            bars1 = ax.bar(x - width, precision_vals, width, label='Precision', color='steelblue')
            bars2 = ax.bar(x, recall_vals, width, label='Recall', color='darkorange')
            bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', color='forestgreen')

            ax.set_xlabel('Class')
            ax.set_ylabel('Score')
            ax.set_title('Per-Class Metrics')
            ax.set_xticks(x)
            ax.set_xticklabels(class_labels, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            # Save and log
            plot_path = f"{name}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(plot_path)
            plt.close()
            Path(plot_path).unlink()

            logger.info(f"Logged per-class metrics: {name}.png")
        except Exception as e:
            logger.warning(f"Failed to log per-class metrics: {e}")

    def log_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        name: str = "classification_report"
    ):
        """
        Save classification report as text artifact.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            labels: Class labels (optional)
            name: Name for the text file
        """
        try:
            # Generate classification report
            report = classification_report(y_true, y_pred, target_names=labels)

            # Save as text file
            report_path = f"{name}.txt"
            with open(report_path, "w") as f:
                f.write("Classification Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(report)

            mlflow.log_artifact(report_path)
            Path(report_path).unlink()

            logger.info(f"Logged classification report: {name}.txt")
        except Exception as e:
            logger.warning(f"Failed to log classification report: {e}")
