"""Build models from config by name, so swapping encoders never touches the trainer.

To add an encoder (e.g. resnet50): write the Encoder subclass in src/models/encoders/,
then add one line to ENCODERS. Its config keys go under model.encoder in the YAML.
"""

from collections.abc import Callable

from src.models.classifier import Classifier
from src.models.encoders.base import Encoder
from src.models.encoders.scratch_cnn import ScratchCNN
from src.models.heads.classification import ClassificationHead
from src.utils.config import ModelConfig

ENCODERS: dict[str, Callable[..., Encoder]] = {
    "scratch_cnn": ScratchCNN,
}


def build_encoder(name: str, **kwargs) -> Encoder:
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}; available: {sorted(ENCODERS)}")
    return ENCODERS[name](**kwargs)


def build_classifier(cfg: ModelConfig, num_classes: int) -> Classifier:
    """Encoder from cfg.name + cfg.encoder, head with cfg.dropout; frozen encoder if cfg.freeze_encoder."""
    encoder = build_encoder(cfg.name, **cfg.encoder)
    if cfg.freeze_encoder:
        encoder.freeze()
    return Classifier(encoder, ClassificationHead(encoder.out_dim, num_classes, cfg.dropout))
