"""Etape 3 : Publication du modele sur HuggingFace Hub."""

from pathlib import Path
from huggingface_hub import HfApi, upload_folder
from flair_lm_trainer.model_card import generate_model_card


def publish(config, model_path):
    model_dir = Path(model_path)

    if config.model_repo_name is None:
        print("Pas de model_repo_name fourni, publication ignoree.")
        return None

    # Utilise le token du config, ou celui deja configure sur la machine
    token = config.hf_token

    print("Publication vers : " + config.model_repo_name)

    # 1. Creer le repo sur HuggingFace
    api = HfApi(token=token)
    api.create_repo(
        repo_id=config.model_repo_name,
        exist_ok=True,
        private=config.private,
    )

    # 2. Generer et sauvegarder la Model Card
    card_text = generate_model_card(config)
    card_path = model_dir / "README.md"
    card_path.write_text(card_text, encoding="utf-8")
    print("Model Card generee.")

    # 3. Upload tout le dossier du modele
    upload_folder(
        repo_id=config.model_repo_name,
        folder_path=str(model_dir),
        token=token,
    )

    url = "https://huggingface.co/" + config.model_repo_name
    print("Modele publie ! " + url)
    return url