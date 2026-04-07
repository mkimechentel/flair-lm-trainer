"""Configuration du pipeline d'entraînement Flair LM."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Tous les paramètres pour le pipeline d'entraînement."""

    source_dataset: str
    text_column: str = "text"
    finetune_from: Optional[str] = None
    is_forward_lm: bool = True
    hidden_size: int = 1024
    nlayers: int = 1
    sequence_length: int = 250
    mini_batch_size: int = 100
    max_epochs: int = 10
    learning_rate: float = 20.0
    model_repo_name: Optional[str] = None
    hf_token: Optional[str] = None
    private: bool = False
    output_dir: str = "output"