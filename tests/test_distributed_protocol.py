# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from types import MappingProxyType, SimpleNamespace
from typing import cast

from heretic.distributed_protocol import (
    LoadAcknowledgement,
    LoadCommand,
    ModelLoadIdentity,
    canonicalize_quantization_config,
    validate_load_acknowledgement,
)


def _valid_identity() -> ModelLoadIdentity:
    return ModelLoadIdentity(
        base_model_id="example/fp8-model",
        base_model_revision="a" * 40,
        dtype="torch.bfloat16",
        quantization_config=(
            '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
            '"quant_method":"compressed-tensors","version":"0.18.0"}'
        ),
    )


class LoadHandshakeTests(unittest.TestCase):
    def test_validator_rechecks_record_and_identity_invariants(self) -> None:
        identity = _valid_identity()
        command = LoadCommand(command_id=41, identity=identity)
        acknowledgement = LoadAcknowledgement(
            command_id=41,
            rank=1,
            identity=identity,
        )
        invalid_identity = SimpleNamespace(
            base_model_id="",
            base_model_revision="main",
            dtype="",
            quantization_config="{}",
        )

        with self.assertRaises(TypeError):
            validate_load_acknowledgement(
                cast(
                    LoadCommand,
                    SimpleNamespace(command_id=41, identity=invalid_identity),
                ),
                cast(
                    LoadAcknowledgement,
                    SimpleNamespace(
                        command_id=41,
                        rank=1,
                        identity=invalid_identity,
                    ),
                ),
            )

        mutated_identity = _valid_identity()
        object.__setattr__(mutated_identity, "base_model_revision", "main")
        with self.assertRaises(ValueError):
            validate_load_acknowledgement(
                LoadCommand(command_id=41, identity=_valid_identity()),
                LoadAcknowledgement(
                    command_id=41,
                    rank=1,
                    identity=mutated_identity,
                ),
            )

        object.__setattr__(command, "command_id", True)
        object.__setattr__(acknowledgement, "command_id", True)
        with self.assertRaises(ValueError):
            validate_load_acknowledgement(command, acknowledgement)

    def test_rejects_quantization_string_subclasses(self) -> None:
        class AlwaysEqualString(str):
            def __eq__(self, other: object) -> bool:
                return True

            def __ne__(self, other: object) -> bool:
                return False

        hostile = AlwaysEqualString(
            '{"config_groups":{"group_0":{"weights":{"num_bits":4}}},'
            '"quant_method":"compressed-tensors"}'
        )
        with self.assertRaisesRegex(TypeError, "quantization_config"):
            ModelLoadIdentity(
                base_model_id="example/fp8-model",
                base_model_revision="a" * 40,
                dtype="torch.bfloat16",
                quantization_config=hostile,
            )

    def test_rejects_duck_typed_identity_objects(self) -> None:
        bypass = SimpleNamespace(
            base_model_id="",
            base_model_revision="main",
            dtype="",
            quantization_config="{}",
        )

        with self.assertRaisesRegex(TypeError, "ModelLoadIdentity"):
            LoadCommand(command_id=1, identity=cast(ModelLoadIdentity, bypass))
        with self.assertRaisesRegex(TypeError, "ModelLoadIdentity"):
            LoadAcknowledgement(
                command_id=1,
                rank=1,
                identity=cast(ModelLoadIdentity, bypass),
            )

    def test_canonicalizes_nested_mapping_implementations(self) -> None:
        config = MappingProxyType(
            {
                "quant_method": "compressed-tensors",
                "config_groups": MappingProxyType(
                    {"group_0": MappingProxyType({"weights": {"num_bits": 8}})}
                ),
            }
        )

        self.assertEqual(
            canonicalize_quantization_config(config),
            (
                '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                '"quant_method":"compressed-tensors"}'
            ),
        )

    def test_rejects_non_integer_protocol_identifiers(self) -> None:
        identity = ModelLoadIdentity(
            base_model_id="example/fp8-model",
            base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            dtype="torch.bfloat16",
            quantization_config=(
                '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                '"quant_method":"compressed-tensors","version":"0.18.0"}'
            ),
        )

        for invalid_command_id in (True, 1.0, -1):
            with self.subTest(command_id=invalid_command_id):
                with self.assertRaisesRegex(ValueError, "command_id"):
                    LoadCommand(
                        command_id=cast(int, invalid_command_id), identity=identity
                    )

        for invalid_command_id in (True, 1.0, -1):
            with self.subTest(acknowledgement_command_id=invalid_command_id):
                with self.assertRaisesRegex(ValueError, "command_id"):
                    LoadAcknowledgement(
                        command_id=cast(int, invalid_command_id),
                        rank=1,
                        identity=identity,
                    )

        for invalid_rank in (True, 1.0, -1):
            with self.subTest(rank=invalid_rank):
                with self.assertRaisesRegex(ValueError, "rank"):
                    LoadAcknowledgement(
                        command_id=1,
                        rank=cast(int, invalid_rank),
                        identity=identity,
                    )

    def test_rejects_empty_or_mutable_checkpoint_identity(self) -> None:
        complete_quantization = (
            '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
            '"quant_method":"compressed-tensors","version":"0.18.0"}'
        )
        valid_fields = {
            "base_model_id": "example/fp8-model",
            "base_model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dtype": "torch.bfloat16",
            "quantization_config": complete_quantization,
        }

        for field_name in ("base_model_id", "base_model_revision", "dtype"):
            with self.subTest(empty_field=field_name):
                values = dict(valid_fields)
                values[field_name] = "   "
                with self.assertRaisesRegex(ValueError, field_name):
                    ModelLoadIdentity(**values)

        for mutable_revision in ("main", "master", "latest", "HEAD", "release-v1"):
            with self.subTest(mutable_revision=mutable_revision):
                values = dict(valid_fields)
                values["base_model_revision"] = mutable_revision
                with self.assertRaisesRegex(ValueError, "pinned"):
                    ModelLoadIdentity(**values)

    def test_rejects_incomplete_quantization_configuration(self) -> None:
        for incomplete_config in (
            "{}",
            '{"quant_method":"compressed-tensors"}',
            '{"config_groups":{}}',
            '{"quant_method":""}',
            '{"config_groups":{},"quant_method":"x"}',
            '{"junk":null,"quant_method":"x"}',
            '{"quant_method":"x","scheme":[]}',
            '{"quant_method":"x","scheme":""}',
            '{"quant_method":"x","version":"1"}',
        ):
            with self.subTest(quantization_config=incomplete_config):
                with self.assertRaisesRegex(ValueError, "complete"):
                    ModelLoadIdentity(
                        base_model_id="example/fp8-model",
                        base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        dtype="torch.bfloat16",
                        quantization_config=incomplete_config,
                    )

    def test_rejects_noncanonical_quantization_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical"):
            ModelLoadIdentity(
                base_model_id="example/fp8-model",
                base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                dtype="torch.bfloat16",
                quantization_config=(
                    '{"version":"0.18.0","quant_method":"compressed-tensors",'
                    '"config_groups":{"group_0":{"weights":{"num_bits":8}}}}'
                ),
            )

    def test_canonicalizes_complete_quantization_configuration(self) -> None:
        first = {
            "version": "0.18.0",
            "config_groups": {
                "group_0": {
                    "weights": {"type": "float", "num_bits": 8},
                    "input_activations": {"dynamic": True, "num_bits": 8},
                }
            },
            "quant_method": "compressed-tensors",
        }
        second = {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "input_activations": {"num_bits": 8, "dynamic": True},
                    "weights": {"num_bits": 8, "type": "float"},
                }
            },
            "version": "0.18.0",
        }

        self.assertEqual(
            canonicalize_quantization_config(first),
            canonicalize_quantization_config(second),
        )

    def test_accepts_exact_rank_one_load_acknowledgement(self) -> None:
        identity = ModelLoadIdentity(
            base_model_id="example/fp8-model",
            base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            dtype="torch.bfloat16",
            quantization_config=(
                '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                '"quant_method":"compressed-tensors","version":"0.18.0"}'
            ),
        )
        command = LoadCommand(command_id=1, identity=identity)
        acknowledgement = LoadAcknowledgement(
            command_id=1,
            rank=1,
            identity=identity,
        )

        validate_load_acknowledgement(command, acknowledgement)

    def test_rejects_load_acknowledgement_from_wrong_rank(self) -> None:
        identity = ModelLoadIdentity(
            base_model_id="example/fp8-model",
            base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            dtype="torch.bfloat16",
            quantization_config=(
                '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                '"quant_method":"compressed-tensors","version":"0.18.0"}'
            ),
        )
        command = LoadCommand(command_id=1, identity=identity)
        acknowledgement = LoadAcknowledgement(
            command_id=1,
            rank=0,
            identity=identity,
        )

        with self.assertRaisesRegex(RuntimeError, "rank"):
            validate_load_acknowledgement(command, acknowledgement)

    def test_rejects_stale_command_acknowledgement(self) -> None:
        identity = ModelLoadIdentity(
            base_model_id="example/fp8-model",
            base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            dtype="torch.bfloat16",
            quantization_config=(
                '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                '"quant_method":"compressed-tensors","version":"0.18.0"}'
            ),
        )
        command = LoadCommand(command_id=2, identity=identity)
        acknowledgement = LoadAcknowledgement(
            command_id=1,
            rank=1,
            identity=identity,
        )

        with self.assertRaisesRegex(RuntimeError, "command_id"):
            validate_load_acknowledgement(command, acknowledgement)

    def test_rejects_base_model_id_mismatch(self) -> None:
        command = LoadCommand(
            command_id=1,
            identity=ModelLoadIdentity(
                base_model_id="example/fp8-model",
                base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                dtype="torch.bfloat16",
                quantization_config=(
                    '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                    '"quant_method":"compressed-tensors","version":"0.18.0"}'
                ),
            ),
        )
        acknowledgement = LoadAcknowledgement(
            command_id=1,
            rank=1,
            identity=ModelLoadIdentity(
                base_model_id="other/fp8-model",
                base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                dtype="torch.bfloat16",
                quantization_config=(
                    '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                    '"quant_method":"compressed-tensors","version":"0.18.0"}'
                ),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "base_model_id"):
            validate_load_acknowledgement(command, acknowledgement)

    def test_rejects_base_revision_mismatch(self) -> None:
        command = LoadCommand(
            command_id=1,
            identity=ModelLoadIdentity(
                base_model_id="example/fp8-model",
                base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                dtype="torch.bfloat16",
                quantization_config=(
                    '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                    '"quant_method":"compressed-tensors","version":"0.18.0"}'
                ),
            ),
        )
        acknowledgement = LoadAcknowledgement(
            command_id=1,
            rank=1,
            identity=ModelLoadIdentity(
                base_model_id="example/fp8-model",
                base_model_revision="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                dtype="torch.bfloat16",
                quantization_config=(
                    '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                    '"quant_method":"compressed-tensors","version":"0.18.0"}'
                ),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "base_model_revision"):
            validate_load_acknowledgement(command, acknowledgement)

    def test_rejects_dtype_mismatch(self) -> None:
        command = LoadCommand(
            command_id=1,
            identity=ModelLoadIdentity(
                base_model_id="example/fp8-model",
                base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                dtype="torch.bfloat16",
                quantization_config=(
                    '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                    '"quant_method":"compressed-tensors","version":"0.18.0"}'
                ),
            ),
        )
        acknowledgement = LoadAcknowledgement(
            command_id=1,
            rank=1,
            identity=ModelLoadIdentity(
                base_model_id="example/fp8-model",
                base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                dtype="torch.float16",
                quantization_config=(
                    '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                    '"quant_method":"compressed-tensors","version":"0.18.0"}'
                ),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "dtype"):
            validate_load_acknowledgement(command, acknowledgement)

    def test_rejects_quantization_configuration_mismatch(self) -> None:
        command = LoadCommand(
            command_id=1,
            identity=ModelLoadIdentity(
                base_model_id="example/fp8-model",
                base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                dtype="torch.bfloat16",
                quantization_config=(
                    '{"config_groups":{"group_0":{"weights":{"num_bits":8}}},'
                    '"quant_method":"compressed-tensors","version":"0.18.0"}'
                ),
            ),
        )
        acknowledgement = LoadAcknowledgement(
            command_id=1,
            rank=1,
            identity=ModelLoadIdentity(
                base_model_id="example/fp8-model",
                base_model_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                dtype="torch.bfloat16",
                quantization_config=(
                    '{"config_groups":{"group_0":{"weights":{"num_bits":4}}},'
                    '"quant_method":"compressed-tensors","version":"0.19.0"}'
                ),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "quantization_config"):
            validate_load_acknowledgement(command, acknowledgement)


if __name__ == "__main__":
    unittest.main()
