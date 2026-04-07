# flair-lm-trainer

Entraine un Flair Language Model (FlairEmbeddings) a partir d'un dataset texte HuggingFace, puis publie optionnellement le modele sur le HuggingFace Hub.

## Installation
```bash
uv add flair-lm-trainer
```

## Utilisation (CLI)
```bash
# Entrainer from scratch
flair-lm-train --dataset "dh-unibe/towerbooks-plaintext"

# Fine-tuner un modele existant
flair-lm-train --dataset "dh-unibe/towerbooks-plaintext" --finetune-from "de-forward"

# Entrainer et publier
flair-lm-train --dataset "dh-unibe/towerbooks-plaintext" --publish --model-repo "mon-org/mon-modele" --private
```

## Utilisation (Python)
```python
from flair_lm_trainer import TrainingConfig, preprocess, train, publish

config = TrainingConfig(
    source_dataset="dh-unibe/towerbooks-plaintext",
    max_epochs=10,
)

corpus_dir = preprocess(config)
model_path = train(config, corpus_dir)
```

## Parametres

Voir `flair-lm-train --help` pour la liste complete des parametres.