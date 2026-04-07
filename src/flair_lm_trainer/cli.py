"""Point d'entree en ligne de commande pour flair-lm-trainer."""

import argparse
from flair_lm_trainer.config import TrainingConfig
from flair_lm_trainer.preprocess import preprocess
from flair_lm_trainer.train import train
from flair_lm_trainer.publish import publish


def main():
    parser = argparse.ArgumentParser(
        description="Entraine un Flair Language Model a partir d'un dataset HuggingFace."
    )

    # Dataset
    parser.add_argument("--dataset", required=True, help="Nom du dataset HuggingFace")
    parser.add_argument("--text-column", default="text", help="Colonne texte (defaut: text)")

    # Modele
    parser.add_argument("--finetune-from", default=None, help="FlairEmbeddings a fine-tuner (ex: de-forward)")
    parser.add_argument("--backward", action="store_true", help="Entrainer un LM backward au lieu de forward")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Taille du hidden layer (defaut: 1024)")
    parser.add_argument("--nlayers", type=int, default=1, help="Nombre de couches (defaut: 1)")

    # Entrainement
    parser.add_argument("--sequence-length", type=int, default=250, help="Longueur des sequences (defaut: 250)")
    parser.add_argument("--mini-batch-size", type=int, default=100, help="Taille des mini-batchs (defaut: 100)")
    parser.add_argument("--max-epochs", type=int, default=10, help="Nombre d'epochs (defaut: 10)")
    parser.add_argument("--learning-rate", type=float, default=20.0, help="Learning rate (defaut: 20)")

    # Publication
    parser.add_argument("--publish", action="store_true", help="Publier le modele sur HuggingFace")
    parser.add_argument("--model-repo", default=None, help="Nom du repo HuggingFace")
    parser.add_argument("--private", action="store_true", help="Rendre le repo prive")

    # Sortie
    parser.add_argument("--output-dir", default="output", help="Dossier de sortie (defaut: output)")

    args = parser.parse_args()

    # Construire la config a partir des arguments
    config = TrainingConfig(
        source_dataset=args.dataset,
        text_column=args.text_column,
        finetune_from=args.finetune_from,
        is_forward_lm=not args.backward,
        hidden_size=args.hidden_size,
        nlayers=args.nlayers,
        sequence_length=args.sequence_length,
        mini_batch_size=args.mini_batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        model_repo_name=args.model_repo,
        private=args.private,
        output_dir=args.output_dir,
    )

    # Etape 1 : Preprocessing
    print("=== PREPROCESSING ===")
    corpus_dir = preprocess(config)

    # Etape 2 : Entrainement
    print("")
    print("=== ENTRAINEMENT ===")
    model_path = train(config, corpus_dir)

    # Etape 3 : Publication (si demandee)
    if args.publish:
        print("")
        print("=== PUBLICATION ===")
        publish(config, model_path)

    print("")
    print("Termine !")


if __name__ == "__main__":
    main()