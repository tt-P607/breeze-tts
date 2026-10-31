from types import SimpleNamespace

import torch
from transformers.generation import GenerationMixin

from models.generation_breeze import BreezeGenerationMixin


def test_empty_audio_codes_use_platform_temporary_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        GenerationMixin,
        "generate",
        lambda self, **kwargs: torch.zeros((1, 1, 2), dtype=torch.long),
    )
    monkeypatch.setattr(
        "models.generation_breeze.tempfile.gettempdir", lambda: str(tmp_path)
    )

    class Model(BreezeGenerationMixin):
        config = SimpleNamespace(num_codebooks=2, codebook_pad_token_id=0)
        codec_model = SimpleNamespace(num_codebooks=2)

    class AudioTokenizer:
        def decode(self, payload):
            assert tuple(payload["audio_codes"].shape) == (1, 2)
            return SimpleNamespace(audio_values=torch.zeros((1, 4)))

    audio = Model().generate(
        input_ids=torch.ones((1, 1), dtype=torch.long),
        output_audio=True,
        audio_tokenizer=AudioTokenizer(),
    )

    warning_log = tmp_path / "breeze_generation_warnings.log"
    assert warning_log.read_text(encoding="utf-8") == (
        "[no-code] No codebook tokens were generated, generating silent audio.\n"
    )
    assert len(audio) == 1
    torch.testing.assert_close(audio[0], torch.zeros(4))
