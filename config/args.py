from dataclasses import dataclass


@dataclass
class Args:
    """
    Training arguments.
    """

    # Learning rate for the optimizer
    learning_rate: float = 1e-3
    # Training batch size
    batch_size: int = 64
    # Total numebr of classes
    num_classes: int = 10
    # Maximum number of training epochs
    max_epochs: int = 100
    # Input shape
    input_shape: tuple = (3, 224, 224)
    # Use pretrained weights
    # Can be "IMAGENET1K_V1", "IMAGENET1K_V2", "DEFAULT"
    # CHec more at https://pytorch.org/vision/stable/models.html
    weights: str = None
