# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging as py_logging
import warnings

import lightning.pytorch as pl

from nemo.collections.tts.models import HifiGanModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


# Quiet noisy dependency logs and SyntaxWarnings from pydub
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub.utils")
for name in ("transformer_engine", "OneLogger", "nemo_logger"):
    py_logging.getLogger(name).setLevel(py_logging.ERROR)


@hydra_runner(config_path="conf", config_name="hifigan")
def main(cfg):
    # Override dataset paths
    cfg.model.train_ds.dataset = None
    cfg.model.validation_ds.dataset = None

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)

    # Manually load datasets
    from nemo.collections.tts.data.vocoder_dataset import VocoderDataset

    train_dataset = VocoderDataset(
        cfg.train_dataset,
        sample_rate=cfg.sample_rate,
        n_segments=cfg.train_n_segments,
        max_duration=cfg.train_max_duration,
        min_duration=cfg.train_min_duration,
        load_precomputed_mel=True,
    )

    val_dataset = VocoderDataset(
        cfg.validation_datasets,
        sample_rate=cfg.sample_rate,
        n_segments=cfg.val_n_segments,
        max_duration=cfg.val_max_duration,
        min_duration=cfg.val_min_duration,
        load_precomputed_mel=True,
    )

    from torch.utils.data import DataLoader

    model.train_dataloader = lambda: DataLoader(
        train_dataset,
        batch_size=cfg.model.train_ds.dataloader_params.batch_size,
        shuffle=True,
        num_workers=cfg.model.train_ds.dataloader_params.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    model.val_dataloader = lambda: [
        DataLoader(
            val_dataset,
            batch_size=cfg.model.validation_ds.dataloader_params.batch_size,
            shuffle=False,
            num_workers=cfg.model.validation_ds.dataloader_params.num_workers,
            pin_memory=True,
        )
    ]


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
