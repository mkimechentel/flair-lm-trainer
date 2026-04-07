"""Etape 2 : Entrainement du Flair Language Model."""

from pathlib import Path
import torch.nn as nn
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair_lm_trainer.config import TrainingConfig


def train(config, corpus_dir):
    corpus_path = Path(corpus_dir)
    output_path = Path(config.output_dir) / "model"

    dictionary = Dictionary.load("chars")

    print("Chargement du corpus...")
    corpus = TextCorpus(
        corpus_path,
        dictionary,
        config.is_forward_lm,
        character_level=True,
    )

    if config.finetune_from is not None:
        print("Chargement du modele existant : " + config.finetune_from)
        emb = FlairEmbeddings(config.finetune_from)
        language_model = emb.lm
        # Recreer le decodeur (Flair le supprime au chargement)
        nout = len(language_model.dictionary)
        language_model.decoder = nn.Linear(language_model.hidden_size, nout)
    else:
        print("Creation d'un nouveau LanguageModel (from scratch)")
        language_model = LanguageModel(
            dictionary,
            config.is_forward_lm,
            hidden_size=config.hidden_size,
            nlayers=config.nlayers,
        )

    trainer = LanguageModelTrainer(language_model, corpus)

    print("Lancement de l'entrainement...")
    trainer.train(
        str(output_path),
        sequence_length=config.sequence_length,
        mini_batch_size=config.mini_batch_size,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        num_workers=0,
    )

    print("Entrainement termine ! Modele sauvegarde dans : " + str(output_path))
    return output_path