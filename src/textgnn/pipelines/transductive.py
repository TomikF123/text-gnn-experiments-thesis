"""
Transductive learning pipeline for TextGCN and other graph-based models.

This pipeline handles models that see all data during training but only
use labels from the training set for supervision. The train/val/test split
determines which node labels are used, not which nodes are visible.
"""

from textgnn.config_class import Config
from textgnn.logger import setup_logger

logger = setup_logger(__name__)


def run_transductive_pipeline(config: Config):
    """
    Run training pipeline for transductive learning models (TextGCN).

    Pipeline:
    1. Load train, val, and test datasets (same graph, different masks)
    2. Create DataLoaders for each split
    3. Create model
    4. Train model with validation (dict of dataloaders)
    5. Evaluate on test set (single graph with test mask)
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
    from textgnn.models.textgcn.eval import eval as textgcn_eval
    from torch.utils.data import DataLoader
    import torch

    logger.info("Running transductive learning pipeline...")

    # Start MLflow tracking
    with ExperimentTracker(config) as tracker:
        # Load datasets (same graph structure, different split masks)
        logger.info("Loading train dataset...")
        train_dataset = load_data(
            dataset_config=config.dataset,
            model_type=config.model_conf.model_type,
            split="train"
        )

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
            shuffle=False  # Single graph, no shuffling needed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=test_dataset.collate_fn,
            shuffle=False
        )

        # Create model
        logger.info("Creating model...")
        model = create_model(
            model_config=config.model_conf,
            dataset_config=config.dataset,
            dataset=train_dataset
        )

        # Log model architecture summary
        tracker.log_model_summary(model)

        # Train model with validation (expects dict of dataloaders)
        logger.info("Training model with validation...")
        dataloaders = {"train": train_loader, "val": val_loader}
        trained_model = train_model(
            model=model,
            dataloaders=dataloaders,
            config=config.model_conf
        )

        # Evaluate on test set (single graph with test mask)
        logger.info("Evaluating on test set...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model.eval()

        test_data = next(iter(test_loader))
        test_data = test_data.to(device)

        with torch.no_grad():
            logits = trained_model(test_data)
            test_nodes = test_data.doc_mask & test_data.split_mask
            y_pred = logits[test_nodes].argmax(dim=1).cpu().numpy()
            y_true = test_data.y[test_nodes].cpu().numpy()

        metrics = textgcn_eval(y_pred, y_true)

        # Log test metrics to MLflow
        test_metrics_prefixed = {f"test_{k}": v for k, v in metrics.items()}
        tracker.log_metrics(test_metrics_prefixed)
        logger.info(f"Test Results: {metrics}")

        # Save model checkpoint
        tracker.log_model(trained_model, model_name="best_model")

        # Log confusion matrix
        tracker.log_confusion_matrix(y_true, y_pred, name="test_confusion_matrix")

    logger.info("Transductive pipeline complete!")
    return trained_model
