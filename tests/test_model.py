# SPDX-License-Identifier: AGPL-3.0-or-later

from transformers import PretrainedConfig

from heretic.model import get_tokenizer_kwargs


def test_laguna_tokenizer_skips_composite_rope_config(monkeypatch) -> None:
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: ({"model_type": "laguna"}, {}),
    )

    kwargs = get_tokenizer_kwargs("poolside/Laguna-S-2.1-FP8", {"revision": "x"})

    assert kwargs["revision"] == "x"
    assert type(kwargs["config"]) is PretrainedConfig
    assert kwargs["fix_mistral_regex"] is True


def test_standard_tokenizer_kwargs_are_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: ({"model_type": "qwen2"}, {}),
    )

    assert get_tokenizer_kwargs("Qwen/Qwen2.5", {"revision": "x"}) == {
        "revision": "x"
    }
