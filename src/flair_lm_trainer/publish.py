"""Step 3: Publish the trained model to the HuggingFace Hub."""

from pathlib import Path
from huggingface_hub import HfApi, upload_folder
from flair_lm_trainer.model_card import generate_model_card


def publish(config, model_path):
    model_dir = Path(model_path)

    if config.model_repo_name is None:
        print("Pas de model_repo_name fourni, publication ignoree.")
        return None

    # Use the token from config, or the one already set up on the machine
    token = config.hf_token

    print("Publication vers : " + config.model_repo_name)

    # 1. Create the repo on HuggingFace
    api = HfApi(token=token)
    api.create_repo(
        repo_id=config.model_repo_name,
        exist_ok=True,
        private=config.private,
    )

    # 2. Generate and save the model card
    card_text = generate_model_card(config)
    card_path = model_dir / "README.md"
    card_path.write_text(card_text, encoding="utf-8")
    print("Model Card generee.")

    # 3. Upload best model + card + logs, but NOT the per-epoch checkpoints
    #    (dh-unibe is short on private HF storage, so we skip epoch_*.pt)
    upload_folder(
        repo_id=config.model_repo_name,
        folder_path=str(model_dir),
        token=token,
        ignore_patterns=["epoch_*.pt"],
    )

    url = "https://huggingface.co/" + config.model_repo_name
    print("Modele publie ! " + url)
    return url
