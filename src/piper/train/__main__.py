import logging

import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint # Import ModelCheckpoint

from .vits.dataset import VitsDataModule
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


class VitsLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.num_symbols", "model.num_symbols")
        parser.link_arguments("model.num_speakers", "data.num_speakers")
        parser.link_arguments("model.sample_rate", "data.sample_rate")
        parser.link_arguments("model.filter_length", "data.filter_length")
        parser.link_arguments("model.hop_length", "data.hop_length")
        parser.link_arguments("model.win_length", "data.win_length")
        parser.link_arguments("model.segment_size", "data.segment_size")


def main():
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False

    # Define your ModelCheckpoint callback
    # You can customize these arguments as needed:
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",  # Directory to save checkpoints
        filename="vits-model-{epoch:02d}-{step:05d}", # Checkpoint filename format
        save_top_k=1,             # Save only the best 1 model based on monitor
        monitor="val_loss",       # Metric to monitor (replace with your actual validation loss metric)
        mode="min",               # "min" for loss, "max" for accuracy/score
        every_n_epochs=10,         # Save every 5 epochs
        # or every_n_steps=1000,   # Save every 1000 training steps
        # or train_time_interval="00:01:00", # Save every 1 minute
        save_last=True            # Always save the last checkpoint
    )

    _cli = VitsLightningCLI(  # noqa: ignore=F841
        VitsModel,
        VitsDataModule,
        trainer_defaults={
            "max_epochs": -1,
            "callbacks": [checkpoint_callback]  # Pass your callback here
        }
    )



# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()