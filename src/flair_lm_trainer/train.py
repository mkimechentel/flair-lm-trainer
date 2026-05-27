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

   # Base dictionary + corpus-specific characters (medieval Low German)
    dictionary = Dictionary.load("chars")
    chars_before = len(dictionary)
    corpus_chars = set()
    for txt_file in corpus_path.rglob("*"):
        if txt_file.is_file():
            corpus_chars.update(txt_file.read_text(encoding="utf-8"))
    for ch in sorted(corpus_chars):
        dictionary.add_item(ch)
    print(f"Dictionary: {chars_before} base chars -> {len(dictionary)} after adding corpus")

    print("Chargement du corpus...")
    corpus = TextCorpus(
        corpus_path,
        dictionary,
        config.is_forward_lm,
        character_level=True,
    )

    if config.finetune_from is not None: 
        print("Chargement du modele existant : " + config.finetune_from)
        emb = FlairEmbeddings(config.finetune_from) # Télécharge le modèle existant
        language_model = emb.lm # Récupère le Language Model dedans
        nout = len(language_model.dictionary) # Compte le nombre de caractères connus
        language_model.decoder = nn.Linear(language_model.hidden_size, nout) # Recreer le decodeur (Flair le supprime au chargement)
    else:
        print("Creation d'un nouveau LanguageModel (from scratch)")
        language_model = LanguageModel( # Crée un modèle vierge
            dictionary, # L'alphabet de caractères
            config.is_forward_lm,  # Forward ou backward
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