"""
Inductive learning pipeline for LSTM and FastText models.

This pipeline handles models that learn from training data and generalize
to unseen test data (standard supervised learning).
"""

from textgnn.config_class import Config
from textgnn.logger import setup_logger

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

    # Start MLflow tracking
    with ExperimentTracker(config) as tracker:
        # Check if validation split exists
        has_validation = config.dataset.tvt_split[1] > 0

        # Load datasets
        logger.info("Loading train dataset...")
        train_dataset = load_data(
            dataset_config=config.dataset,
            model_type=config.model_conf.model_type,
            split="train"
        )

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            num_workers=0  # Single process (GloVe can't be pickled for multiprocessing)
        )

        if has_validation:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                collate_fn=val_dataset.collate_fn,
                shuffle=False,
                num_workers=0
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=test_dataset.collate_fn,
            shuffle=False,
            num_workers=0
        )

        # Create model
        logger.info("Creating model...")
        model = create_model(
            model_config=config.model_conf,
            dataset_config=config.dataset
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
        test_metrics, y_true, y_pred = evaluate(
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

        # Log confusion matrix
        tracker.log_confusion_matrix(y_true, y_pred, name="test_confusion_matrix")

    logger.info("Inductive pipeline complete!")
    return trained_model
