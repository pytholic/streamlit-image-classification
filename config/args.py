from dataclasses import dataclass


@dataclass
class Args:
    """
    Training arguments.
    """

    # Learning rate for the optimizer
    learning_rate: float = 1e-3
    # Training batch size
    batch_size: int = 32
    # Total numebr of classes
    num_classes: int = 10
    # Maximum number of training epochs
    max_epochs: int = 5
