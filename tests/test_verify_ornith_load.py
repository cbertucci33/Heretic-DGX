# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "verify-ornith-load.py"
SPEC = importlib.util.spec_from_file_location("verify_ornith_load", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load verifier module: {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QuantizationMetadataTests(unittest.TestCase):
    def test_reads_quant_method_from_mapping(self) -> None:
        config = SimpleNamespace(quantization_config={"quant_method": "compressed-tensors"})

        self.assertEqual(MODULE.get_quant_method(config), "compressed-tensors")

    def test_reads_quant_method_from_config_object(self) -> None:
        quantization_config = SimpleNamespace(quant_method="compressed-tensors")
        config = SimpleNamespace(quantization_config=quantization_config)

        self.assertEqual(MODULE.get_quant_method(config), "compressed-tensors")

    def test_returns_none_when_quantization_metadata_is_absent(self) -> None:
        self.assertIsNone(MODULE.get_quant_method(SimpleNamespace()))


class OrnithIdentityValidationTests(unittest.TestCase):
    def test_requires_exact_ornith_fp8_identity(self) -> None:
        valid = {
            "model_class": "Qwen3_5ForConditionalGeneration",
            "model_type": "qwen3_5",
            "quant_method": "compressed-tensors",
            "parameter_devices": {"cuda:0": 960},
            "parameter_dtypes": {
                "torch.bfloat16": 760,
                "torch.float8_e4m3fn": 200,
            },
        }
        MODULE.validate_loaded_identity(valid)

        mismatches = {
            "model_class": "WrongModel",
            "model_type": "wrong_type",
            "quant_method": None,
            "parameter_devices": {"cpu": 960},
            "parameter_dtypes": {"torch.bfloat16": 960},
        }
        for field, wrong_value in mismatches.items():
            with self.subTest(field=field):
                invalid = copy.deepcopy(valid)
                invalid[field] = wrong_value
                with self.assertRaisesRegex(RuntimeError, field):
                    MODULE.validate_loaded_identity(invalid)


class OrnithWeightValidationTests(unittest.TestCase):
    def test_requires_exact_weight_size_and_sha256(self) -> None:
        payload = b"ornith"
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            weight_path = Path(directory) / "model.safetensors"
            weight_path.write_bytes(payload)

            MODULE.validate_weight_file(
                weight_path,
                expected_size=len(payload),
                expected_sha256=expected_sha256,
            )
            with self.assertRaisesRegex(RuntimeError, "size"):
                MODULE.validate_weight_file(
                    weight_path,
                    expected_size=len(payload) + 1,
                    expected_sha256=expected_sha256,
                )
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                MODULE.validate_weight_file(
                    weight_path,
                    expected_size=len(payload),
                    expected_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
