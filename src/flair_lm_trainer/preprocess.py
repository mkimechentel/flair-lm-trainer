"""Etape 1 : Telechargement du dataset et preparation pour Flair."""

from pathlib import Path
from datasets import load_dataset
from flair_lm_trainer.config import TrainingConfig

NEWLINE = "\n"


def preprocess(config):
    corpus_dir = Path(config.output_dir) / "corpus"
    train_dir = corpus_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    print("Telechargement du dataset...")
    ds = load_dataset(config.source_dataset, split="train")

    texts = ds[config.text_column]
    total = len(texts)
    print(str(total) + " pages chargees.")

    if total < 3:
        print("ERREUR : il faut au moins 3 pages pour decouper en train/valid/test")
        raise ValueError("Dataset trop petit (minimum 3 pages)")

    test_size = max(1, int(total * 0.1))
    valid_size = max(1, int(total * 0.1))
    train_size = total - valid_size - test_size

    train_texts = texts[:train_size]
    valid_texts = texts[train_size:train_size + valid_size]
    test_texts = texts[train_size + valid_size:]

    print("Decoupage : " + str(len(train_texts)) + " train, " + str(len(valid_texts)) + " valid, " + str(len(test_texts)) + " test")

    train_path = train_dir / "train_split_1"
    train_path.write_text(NEWLINE.join(train_texts), encoding="utf-8")

    valid_path = corpus_dir / "valid.txt"
    valid_path.write_text(NEWLINE.join(valid_texts), encoding="utf-8")

    test_path = corpus_dir / "test.txt"
    test_path.write_text(NEWLINE.join(test_texts), encoding="utf-8")

    print("Corpus prepare dans : " + str(corpus_dir))
    return corpus_dir