"""Logging configuration for the textgnn package."""
import logging
import sys
import torch


def setup_logger(name: str = "textgnn", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format: [INFO] textgnn.prep_data: Removing stopwords...
    formatter = logging.Formatter(
        fmt='[%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def log_batch_info(batch, batch_idx=None, epoch=None, logger=None, device=None):
    """
    Log CONCISE information about batch structure and tensor shapes.
    Model-agnostic - works with any batch format.

    Args:
        batch: The batch to inspect (dict, tensor, PyG Data, etc.)
        batch_idx: Batch index (optional)
        epoch: Epoch number (optional)
        logger: Logger instance to use (optional)
        device: Device the batch is on (optional, for display)
    """
    if logger is None:
        logger = setup_logger(__name__)

    # Compact header - all on one line
    if batch_idx is not None:
        if epoch is not None:
            header = f"[BATCH {batch_idx} Epoch {epoch}]"
        else:
            header = f"[BATCH {batch_idx}]"
    else:
        header = "[BATCH]"

    # Add device and batch info to header line
    device_str = f"Device: {device}" if device else ""

    def describe_concise(name, tensor):
        """Concisely describe tensor - single line per item."""
        if isinstance(tensor, torch.Tensor):
            if tensor.is_sparse:
                nnz = tensor._nnz()
                return f"{name}: Sparse {tuple(tensor.shape)} with {nnz:,} nnz"
            else:
                return f"{name}: {tuple(tensor.shape)} {tensor.dtype} on {tensor.device}"

        elif isinstance(tensor, list):
            if len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
                # Show count and first example only
                first = tensor[0]
                if first.is_sparse:
                    nnz = first._nnz()
                    return f"{name}: List[{len(tensor)} sparse tensors] - Example: {tuple(first.shape)} with {nnz} nnz"
                else:
                    return f"{name}: List[{len(tensor)} tensors] - Example: {tuple(first.shape)}"
            else:
                return f"{name}: List[{len(tensor)} items]"

        elif isinstance(tensor, dict):
            # Collect keys count for header
            return f"Dict[{len(tensor)}]"

        return None

    # Get batch summary for header
    batch_summary = describe_concise("batch", batch)

    # Single header line with all info
    parts = [header]
    if device_str:
        parts.append(device_str)
    if batch_summary:
        parts.append(batch_summary)

    logger.info(" | ".join(parts))

    # Now log individual items if batch is a dict
    if isinstance(batch, dict):
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                if val.is_sparse:
                    nnz = val._nnz()
                    logger.info(f"  {key}: Sparse {tuple(val.shape)} with {nnz:,} nnz")
                else:
                    logger.info(f"  {key}: {tuple(val.shape)} {val.dtype} on {val.device}")
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                first = val[0]
                if first.is_sparse:
                    nnz = first._nnz()
                    logger.info(f"  {key}: List[{len(val)} sparse tensors] - Example: {tuple(first.shape)} with {nnz} nnz")
                else:
                    logger.info(f"  {key}: List[{len(val)} tensors] - Example: {tuple(first.shape)}")
        return  # Early return for dict case

    # PyG Data case
    if hasattr(batch, '__dict__'):
        for attr_name in ['x', 'edge_index', 'edge_attr', 'y', 'batch', 'doc_mask', 'split_mask']:
            if hasattr(batch, attr_name):
                try:
                    attr = getattr(batch, attr_name)
                    if isinstance(attr, torch.Tensor):
                        if attr.is_sparse:
                            nnz = attr._nnz()
                            logger.info(f"  {attr_name}: Sparse {tuple(attr.shape)} with {nnz:,} nnz")
                        else:
                            logger.info(f"  {attr_name}: {tuple(attr.shape)} {attr.dtype} on {attr.device}")
                except:
                    pass
