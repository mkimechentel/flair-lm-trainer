"""Generation de la ModelCard pour le modele publie sur HuggingFace."""


def generate_model_card(config):
    mode = "from scratch"
    if config.finetune_from is not None:
        mode = "fine-tuned from " + config.finetune_from

    direction = "forward"
    if not config.is_forward_lm:
        direction = "backward"

    card = "---\n"
    card = card + "tags:\n"
    card = card + "- flair\n"
    card = card + "- language-model\n"
    card = card + "- character-lm\n"
    card = card + "---\n\n"
    card = card + "# Flair Language Model (" + direction + ")\n\n"
    card = card + "## Overview\n\n"
    card = card + "This is a Flair character-level language model (" + direction + ") trained " + mode + ".\n\n"
    card = card + "## Training data\n\n"
    card = card + "Source dataset: `" + config.source_dataset + "`\n\n"
    card = card + "## Training parameters\n\n"
    card = card + "- Hidden size: " + str(config.hidden_size) + "\n"
    card = card + "- Layers: " + str(config.nlayers) + "\n"
    card = card + "- Sequence length: " + str(config.sequence_length) + "\n"
    card = card + "- Mini batch size: " + str(config.mini_batch_size) + "\n"
    card = card + "- Max epochs: " + str(config.max_epochs) + "\n"
    card = card + "- Learning rate: " + str(config.learning_rate) + "\n\n"
    card = card + "## Usage\n\n"
    card = card + "```python\n"
    card = card + "from flair.embeddings import FlairEmbeddings\n\n"
    card = card + "embeddings = FlairEmbeddings('" + str(config.model_repo_name) + "')\n"
    card = card + "```\n"

    return card