"""
Inductive learning pipeline for LSTM and FastText models.

This pipeline handles models that learn from training data and generalize
to unseen test data (standard supervised learning).
"""

from textgnn.config_class import Config
from textgnn.logger import setup_logger
import numpy as np

logger = setup_logger(__name__)


def run_inductive_pipeline(config: Config):
    """
    Run training pipeline for inductive learning models (LSTM, FastText).

    Pipeline:
    1. Load train, validation (optional), and test datasets
    2. Create DataLoaders
    3. Create model
    4. Train model (with validation if available)
    5. Evaluate on test set (batch iteration)
    6. Log all metrics and artifacts to MLflow

    Args:
        config: Pydantic Config object with dataset and model configs

    Returns:
        Trained model
    """
    from textgnn.experiment_tracker import ExperimentTracker
    from textgnn.load_data import load_data
    from textgnn.model_factory import create_model
    from textgnn.train import train_model
    from textgnn.eval import evaluate
    from torch.utils.data import DataLoader
    import torch

    logger.info("Running inductive learning pipeline...")

    # ===== Data Loading Phase (NOT tracked by MLflow) =====
    logger.info("Loading datasets (preprocessing phase - not tracked)...")

    # Check if validation split exists
    has_validation = config.dataset.tvt_split[1] > 0

    # Load datasets
    logger.info("Loading train dataset...")
    train_dataset = load_data(
        dataset_config=config.dataset,
        model_type=config.model_conf.model_type,
        split="train"
    )

    # Subsample training data (for data efficiency experiments)
    if config.dataset.train_subsample is not None and config.dataset.train_subsample < 1.0:
        import random
        frac = config.dataset.train_subsample
        n_original = len(train_dataset)
        n_samples = max(1, int(n_original * frac))
        rng = random.Random(config.dataset.random_seed)
        indices = sorted(rng.sample(range(n_original), n_samples))

        # Subsample all list-like attributes
        if hasattr(train_dataset, 'df'):
            train_dataset.df = train_dataset.df.iloc[indices].reset_index(drop=True)
            train_dataset.texts = train_dataset.df["text"].tolist()
            train_dataset.labels = train_dataset.df["label"].tolist()
        elif hasattr(train_dataset, 'word_ids'):
            train_dataset.word_ids = [train_dataset.word_ids[i] for i in indices]
            train_dataset.adjacencies = [train_dataset.adjacencies[i] for i in indices]
            train_dataset.labels = [train_dataset.labels[i] for i in indices]

        logger.info(f"Subsampled training data: {n_original} -> {n_samples} ({frac*100:.0f}%)")

    if has_validation:
        logger.info("Loading validation dataset...")
        val_dataset = load_data(
            dataset_config=config.dataset,
            model_type=config.model_conf.model_type,
            split="val"
        )

    logger.info("Loading test dataset...")
    test_dataset = load_data(
        dataset_config=config.dataset,
        model_type=config.model_conf.model_type,
        split="test"
    )

    # Create DataLoaders
    logger.info("Creating DataLoaders...")
    batch_size = config.model_conf.common_params["batch_size"]

    # Try multiprocessing for data loading (now that we cache PyTorch sparse tensors)
    # Falls back to 0 if sparse tensors can't be pickled
    num_workers = 4 if torch.cuda.is_available() else 0  # Only if GPU available

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )

    if has_validation:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    logger.info("Data loading complete. Starting MLflow tracking for training phase...")

    # ===== Training & Evaluation Phase (TRACKED by MLflow) =====
    # Start MLflow tracking AFTER data loading to track only training metrics
    with ExperimentTracker(config) as tracker:
        # Create model
        logger.info("Creating model...")
        model = create_model(
            model_config=config.model_conf,
            dataset_config=config.dataset,
            dataset=train_dataset
        )

        # Log model architecture summary
        tracker.log_model_summary(model)

        # Train model (with or without validation)
        if has_validation:
            logger.info("Training model with validation...")
            dataloaders = {"train": train_loader, "val": val_loader}
        else:
            logger.info("Training model without validation...")
            dataloaders = train_loader

        trained_model = train_model(
            model=model,
            dataloaders=dataloaders,
            config=config.model_conf
        )

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_metrics, y_true, y_pred, y_probs = evaluate(
            model=trained_model,
            data_loader=test_loader,
            device=device,
            return_preds=True
        )

        # Log test metrics to MLflow
        test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
        tracker.log_metrics(test_metrics_prefixed)
        logger.info(f"Test Results: {test_metrics}")

        # Save model checkpoint
        tracker.log_model(trained_model, model_name="best_model")

        # Log evaluation artifacts
        tracker.log_confusion_matrix(y_true, y_pred, name="test_confusion_matrix")
        
        # Safely log ROC/PR curves and metrics
        if y_probs is not None and np.unique(y_true).size > 1:
            tracker.log_roc_curve(y_true, y_probs, name="test_roc_curve")
            tracker.log_precision_recall_curve(y_true, y_probs, name="test_pr_curve")

        tracker.log_per_class_metrics(y_true, y_pred, name="test_per_class_metrics")
        tracker.log_classification_report(y_true, y_pred, name="test_classification_report")

        # Save per-document predictions for analysis
        try:
            import mlflow, os
            pred_path = "test_predictions.npz"
            np.savez(pred_path, y_true=y_true, y_pred=y_pred)
            mlflow.log_artifact(pred_path)
            os.remove(pred_path)
        except Exception:
            pass

    logger.info("Inductive pipeline complete!")
    return trained_model
