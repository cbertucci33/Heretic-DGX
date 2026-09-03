# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence

from .rank_environment import RankEnvironment, read_rank_environment


def _validate_application_argv(argv: Sequence[str]) -> None:
    seed_count = 0
    for index, argument in enumerate(argv):
        if argument == "--cluster" or argument.startswith("--cluster="):
            raise ValueError("rank application argv must not contain --cluster")
        if argument == "--seed":
            if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
                raise ValueError("rank application --seed requires a value")
            seed_count += 1
        elif argument.startswith("--seed="):
            if argument == "--seed=":
                raise ValueError("rank application --seed requires a value")
            seed_count += 1
    if seed_count != 1:
        raise ValueError("rank application argv must contain exactly one seed")


def run_rank_application(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    application_main: Callable[[], int | None],
) -> tuple[RankEnvironment, int]:
    """Validate one rank before invoking the upstream application entry."""

    rank_environment = read_rank_environment(environment)
    application_argv = tuple(argv)
    _validate_application_argv(application_argv)
    sys.argv = [sys.argv[0], *application_argv]
    result = application_main()
    if result is None:
        result = 0
    if type(result) is not int:
        raise TypeError("rank application main must return an integer or None")
    return rank_environment, result


def main(argv: list[str] | None = None) -> int:
    application_argv = sys.argv[1:] if argv is None else argv

    # Importing the upstream entry loads the ML runtime, so keep it after the
    # strict environment and recursion checks above.
    def application_main() -> int | None:
        from .main import main as upstream_main

        return upstream_main()

    _, result = run_rank_application(
        application_argv,
        environment=os.environ,
        application_main=application_main,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
