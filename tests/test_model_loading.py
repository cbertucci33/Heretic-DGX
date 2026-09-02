# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch
from types import SimpleNamespace

from heretic.config import Settings
from heretic.model import Model, build_model_load_kwargs, get_model_class


def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr("sys.argv", ["heretic"])
    return Settings(
        model="org/model-under-test",
        model_commit="0123456789abcdef0123456789abcdef01234567",
        device_map="auto",
        max_memory={"0": "100GB", "cpu": "20GB"},
    )


def test_model_class_resolution_uses_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, dict[str, object]]] = []

    def get_config_dict(model: str, **kwargs: object):
        observed.append((model, kwargs))
        return ({"architectures": ["Qwen2ForCausalLM"]}, {})

    monkeypatch.setattr(
        "heretic.model.PretrainedConfig.get_config_dict",
        get_config_dict,
    )

    get_model_class("org/model-under-test", revision="a" * 40)

    assert observed == [("org/model-under-test", {"revision": "a" * 40})]


def test_local_model_load_kwargs_preserve_accelerate_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = build_model_load_kwargs(
        settings(monkeypatch),
        dtype="bfloat16",
        quantization_config=None,
        distributed=False,
        trust_remote_code=False,
    )

    assert kwargs == {
        "dtype": "bfloat16",
        "device_map": "auto",
        "max_memory": {0: "100GB", "cpu": "20GB"},
        "trust_remote_code": None,
        "revision": "0123456789abcdef0123456789abcdef01234567",
    }


def test_dgx_model_load_kwargs_use_transformers_tensor_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = build_model_load_kwargs(
        settings(monkeypatch),
        dtype="bfloat16",
        quantization_config=None,
        distributed=True,
        trust_remote_code=False,
    )

    assert kwargs == {
        "dtype": "bfloat16",
        "tp_plan": "auto",
        "trust_remote_code": None,
        "revision": "0123456789abcdef0123456789abcdef01234567",
    }
    assert "device_map" not in kwargs
    assert "max_memory" not in kwargs


def test_model_constructor_uses_dgx_tp_loading_when_launched_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_kwargs: list[dict[str, object]] = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> "FakeTokenizer":
            return cls()

    class FakeLoadedModel:
        dtype = torch.bfloat16

    class FakeFactory:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> FakeLoadedModel:
            recorded_kwargs.append(kwargs)
            return FakeLoadedModel()

    monkeypatch.setenv("HERETIC_DGX_ACTIVE", "1")
    monkeypatch.setattr("heretic.model.AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(
        "heretic.model.get_model_class",
        lambda model, **kwargs: FakeFactory,
    )
    monkeypatch.setattr(Model, "generate", lambda *args, **kwargs: None)
    monkeypatch.setattr(Model, "_apply_lora", lambda self: None)
    monkeypatch.setattr(Model, "get_layers", lambda self: [])

    Model(settings(monkeypatch))

    assert len(recorded_kwargs) == 1
    assert recorded_kwargs[0]["tp_plan"] == "auto"
    assert "device_map" not in recorded_kwargs[0]
    assert "max_memory" not in recorded_kwargs[0]


def test_dgx_slow_reset_reuses_tensor_parallel_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_kwargs: list[dict[str, object]] = []

    class FakeFactory:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            recorded_kwargs.append(kwargs)
            return object()

    model = object.__new__(Model)
    model.settings = settings(monkeypatch)
    model.model = SimpleNamespace(config=SimpleNamespace(name_or_path="org/old-model"))
    model.dtype = torch.bfloat16
    model.distributed = True
    model.needs_reload = True
    model.trusted_models = set()
    model.revision_kwargs = {"revision": model.settings.model_commit}
    model.max_memory = {0: "100GB", "cpu": "20GB"}

    monkeypatch.setattr("heretic.model.empty_cache", lambda: None)
    monkeypatch.setattr(
        "heretic.model.get_model_class",
        lambda name, **kwargs: FakeFactory,
    )
    monkeypatch.setattr(Model, "_get_quantization_config", lambda self, dtype: None)
    monkeypatch.setattr(Model, "_apply_lora", lambda self: None)

    model.reset_model()

    assert recorded_kwargs[0]["tp_plan"] == "auto"
    assert "device_map" not in recorded_kwargs[0]
    assert "max_memory" not in recorded_kwargs[0]
