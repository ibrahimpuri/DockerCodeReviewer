"""
Fine-tune the UniXcoder classification head on collected training samples.

Design constraints:
  - No HuggingFace Trainer / accelerate dependency.
  - Manual AdamW loop, 3 epochs, gradient clipping at 1.0.
  - Saves with model.save_pretrained() + tokenizer.save_pretrained().
  - Raises ValueError if fewer than MIN_SAMPLES samples are available.
"""

import logging
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from database import get_training_samples

logger = logging.getLogger(__name__)

MODEL_WEIGHTS_DIR = os.getenv("MODEL_WEIGHTS_DIR", "./data/model_weights")

EPOCHS        = 3
BATCH_SIZE    = 8
LEARNING_RATE = 2e-5
MAX_LENGTH    = 512
GRAD_CLIP     = 1.0
MIN_SAMPLES   = 20


class CodeDataset(Dataset):
    """Torch Dataset wrapping pre-tokenized code samples."""

    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def fine_tune(model, tokenizer, device: torch.device) -> dict:
    """
    Fine-tune `model` on all training_samples rows in the database.

    Parameters
    ----------
    model     : AutoModelForSequenceClassification already on `device`
    tokenizer : matching AutoTokenizer
    device    : torch.device

    Returns
    -------
    dict: epochs, samples, final_loss, weights_dir

    Raises
    ------
    ValueError if fewer than MIN_SAMPLES labelled samples exist
    """
    rows = get_training_samples()
    if len(rows) < MIN_SAMPLES:
        raise ValueError(
            f"Fine-tuning requires at least {MIN_SAMPLES} labelled samples; "
            f"only {len(rows)} found. Keep submitting code and providing feedback."
        )

    texts  = [r["code"]  for r in rows]
    labels = [r["label"] for r in rows]
    logger.info("Fine-tuning on %d samples for %d epochs.", len(rows), EPOCHS)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    dataset    = CodeDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer  = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Apply Intel IPEX optimisation when running on an Intel XPU.
    # This is a no-op on all other devices and silently skipped if IPEX is not installed.
    if device.type == "xpu":
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
            logger.info("Intel IPEX optimisation applied.")
        except ImportError:
            pass

    model.train()

    final_loss = 0.0
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / max(len(dataloader), 1)
        final_loss = avg
        logger.info("Epoch %d/%d — avg loss: %.4f", epoch, EPOCHS, avg)

    model.eval()

    os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
    model.save_pretrained(MODEL_WEIGHTS_DIR)
    tokenizer.save_pretrained(MODEL_WEIGHTS_DIR)
    logger.info("Fine-tuned weights saved to %s", MODEL_WEIGHTS_DIR)

    return {
        "epochs":      EPOCHS,
        "samples":     len(rows),
        "final_loss":  round(final_loss, 4),
        "weights_dir": MODEL_WEIGHTS_DIR,
    }
