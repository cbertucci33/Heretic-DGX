import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify-wsl.sh"


def python_heredocs() -> list[str]:
    script = VERIFY_SCRIPT.read_text(encoding="utf-8")
    return re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)


class VerifyWslFailClosedTests(unittest.TestCase):
    def test_baseline_distinguishes_captured_evidence_from_latest_run(self) -> None:
        report_path = (
            REPO_ROOT / "artifacts" / "baseline" / "wsl-cuda-environment-report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

        baseline = report["source_commit_baseline"]
        self.assertEqual(baseline["source_commit"], report["source_commit"])
        self.assertEqual(baseline["captured_at"], report["captured_at"])
        self.assertEqual(baseline["unit_tests"]["count"], 10)
        self.assertIn("Ran 10 tests", baseline["unit_tests"]["observed"])

        latest = report["latest_exact_tree_verifier_run"]
        self.assertEqual(latest["status"], "passed")
        self.assertEqual(
            latest["source_commit"],
            "5fba235708d8b7400abcb7a9c909ed04b290e7f0",
        )
        self.assertEqual(
            latest["git_tree"],
            "aa16b9c6dd10684366681de427aadef87e8cb1f0",
        )
        self.assertEqual(latest["unit_test_count"], 15)
        self.assertIsNotNone(latest["verified_at"])
        self.assertEqual(latest["python_optimize"], "1")
        self.assertEqual(
            latest["evidence_artifact"],
            "artifacts/verification/wsl-reproducibility-workflow-report.json",
        )

    def test_wsl1_kernel_identity_is_not_accepted_as_wsl2(self) -> None:
        script = VERIFY_SCRIPT.read_text(encoding="utf-8")
        gate = script.split("repo_root=", maxsplit=1)[0]

        self.assertIn("microsoft-standard-WSL2", gate)
        self.assertNotRegex(gate, r"grep\s+-qi\s+microsoft\s")

    def test_cuda_gate_fails_when_python_optimization_disables_asserts(self) -> None:
        cuda_check = python_heredocs()[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_torch = Path(temp_dir) / "torch.py"
            fake_torch.write_text(
                textwrap.dedent(
                    """
                    __version__ = "test"

                    class version:
                        cuda = None

                    class cuda:
                        @staticmethod
                        def is_available():
                            return False

                        @staticmethod
                        def device_count():
                            return 0

                        @staticmethod
                        def get_device_name(index):
                            return "none"

                        @staticmethod
                        def get_device_capability(index):
                            return (0, 0)

                        @staticmethod
                        def synchronize():
                            return None

                    class Value:
                        def __mul__(self, other):
                            return self

                        def sum(self):
                            return self

                        def item(self):
                            return 56.0

                    float32 = object()

                    def arange(*args, **kwargs):
                        return Value()
                    """
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            result = subprocess.run(
                [sys.executable, "-O", "-c", cuda_check],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("cuda_available", result.stderr)

    def test_package_gate_fails_when_python_optimization_disables_asserts(self) -> None:
        package_check = python_heredocs()[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir)
            for wheel_name in ("first.whl", "second.whl"):
                with zipfile.ZipFile(build_dir / wheel_name, "w") as archive:
                    archive.writestr("heretic/runtime.py", "")
            (build_dir / "package.tar.gz").write_bytes(b"placeholder")

            result = subprocess.run(
                [sys.executable, "-O", "-c", package_check, temp_dir],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("exactly one wheel", result.stderr)


if __name__ == "__main__":
    unittest.main()
