# SPDX-License-Identifier: AGPL-3.0-or-later


def require_local_model_operation(*, distributed: bool, operation: str) -> None:
    if distributed:
        raise RuntimeError(
            f"{operation} is not available for DGX distributed models until "
            "the operation is implemented collectively"
        )


def require_distributed_model_revision(model: str, revision: str | None) -> None:
    """Require an immutable Hugging Face commit for a distributed model load."""
    if (
        revision is None
        or len(revision) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
    ):
        raise RuntimeError(
            f"DGX distributed model {model!r} requires a 40-character hexadecimal "
            "Hugging Face commit revision"
        )


def raise_if_distributed_failure(*, distributed: bool, error: Exception) -> None:
    """Make rank-local action recovery fatal after a distributed failure."""
    if distributed:
        raise error
