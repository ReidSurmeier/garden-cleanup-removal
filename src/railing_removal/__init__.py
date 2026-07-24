from __future__ import annotations

from typing import Any


__all__ = ["LineCompletionParameters", "complete_railing_lines"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from railing_removal.completion import (
            LineCompletionParameters,
            complete_railing_lines,
        )

        return {
            "LineCompletionParameters": LineCompletionParameters,
            "complete_railing_lines": complete_railing_lines,
        }[name]
    raise AttributeError(name)
