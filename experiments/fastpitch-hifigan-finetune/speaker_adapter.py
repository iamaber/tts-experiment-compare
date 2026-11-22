from contextlib import contextmanager

import torch.nn as nn


class SpeakerAdaptationLayer(nn.Module):
    def __init__(self, input_embedding_dim, model_hidden_dim):
        super().__init__()
        self.projection = nn.Linear(input_embedding_dim, model_hidden_dim)

        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)

    def forward(self, text_hidden_states, speaker_embedding):
        projected_embedding = self.projection(speaker_embedding)

        seq_len = text_hidden_states.size(1)
        expanded_embedding = projected_embedding.unsqueeze(1).expand(-1, seq_len, -1)

        return text_hidden_states + expanded_embedding


@contextmanager
def speaker_conditioned(fastpitch, adapter, speaker_embedding):
    device = next(adapter.parameters()).device
    if speaker_embedding.dim() == 1:
        speaker_embedding = speaker_embedding.unsqueeze(0)
    speaker_embedding = speaker_embedding.to(device)

    def condition_encoder_output(module, inputs, output):
        enc_out, enc_mask = output
        return adapter(enc_out, speaker_embedding), enc_mask

    hook = fastpitch.fastpitch.encoder.register_forward_hook(condition_encoder_output)
    try:
        yield
    finally:
        hook.remove()
