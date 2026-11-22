from nemo.collections.tts.data.dataset import VocoderDataset


class PrecomputedMelVocoderDataset(VocoderDataset):
    """NeMo 2.6's HifiGanModel builds its dataloader with `dataset.collate_fn`,
    but the precomputed-mel VocoderDataset only defines `_collate_fn`."""

    def collate_fn(self, batch):
        return self._collate_fn(batch)
