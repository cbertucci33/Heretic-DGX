# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from pydantic import ValidationError

from heretic.config import ScorerConfig, Settings


def test_explicit_config_path_loads_toml(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config = tmp_path / "selected.toml"
    config.write_text('model = "from-explicit-config"\nseed = 17\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["heretic", "--config", str(config)])

    settings = Settings()  # ty:ignore[missing-argument]

    assert settings.model == "from-explicit-config"
    assert settings.seed == 17


def test_default_config_path_remains_config_toml(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    (tmp_path / "config.toml").write_text('model = "from-default-config"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["heretic"])

    settings = Settings()  # ty:ignore[missing-argument]

    assert settings.model == "from-default-config"


def test_cli_values_override_explicit_config(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config = tmp_path / "selected.toml"
    config.write_text('model = "from-config"\nseed = 17\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["heretic", "--config", str(config), "--model", "from-cli"],
    )

    settings = Settings()  # ty:ignore[missing-argument]

    assert settings.model == "from-cli"
    assert settings.seed == 17


def test_last_duplicate_config_path_selects_matching_toml(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text('model = "from-first"\n')
    second.write_text('model = "from-second"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["heretic", "--config", str(first), "--config", str(second)],
    )

    settings = Settings()  # ty:ignore[missing-argument]

    assert settings.config == str(second)
    assert settings.model == "from-second"


def test_config_equals_supports_dash_prefixed_relative_path(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config = tmp_path / "-selected.toml"
    config.write_text('model = "from-dash-path"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["heretic", "--config=-selected.toml"])

    settings = Settings()  # ty:ignore[missing-argument]

    assert settings.config == "-selected.toml"
    assert settings.model == "from-dash-path"


def test_missing_config_value_fails_closed(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["heretic", "--config", "--model", "x"])

    with pytest.raises(SystemExit) as error:
        Settings()  # ty:ignore[missing-argument]

    assert error.value.code == 2


def test_empty_config_value_fails_closed(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["heretic", "--config="])

    with pytest.raises(ValueError, match="--config requires a non-empty path"):
        Settings()  # ty:ignore[missing-argument]


class ScorerConfigTests(unittest.TestCase):
    def test_accepts_slug_like_instance_name(self) -> None:
        config = ScorerConfig(
            plugin="heretic.scorers.keyword_rate.KeywordRate",
            optimization="minimize",
            instance_name="small-1",
        )

        self.assertEqual(config.instance_name, "small-1")

    def test_rejects_empty_instance_name(self) -> None:
        with self.assertRaises(ValidationError):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name=" \t",
            )

    def test_rejects_whitespace_in_instance_name(self) -> None:
        for instance_name in ["small name", "small\tname", "small\nname"]:
            with self.subTest(instance_name=instance_name):
                with self.assertRaisesRegex(
                    ValidationError, "whitespace is not allowed"
                ):
                    ScorerConfig(
                        plugin="heretic.scorers.keyword_rate.KeywordRate",
                        optimization="minimize",
                        instance_name=instance_name,
                    )

    def test_rejects_dot_in_instance_name(self) -> None:
        with self.assertRaisesRegex(ValidationError, "'\\.' is not allowed"):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name="small.name",
            )


if __name__ == "__main__":
    unittest.main()
