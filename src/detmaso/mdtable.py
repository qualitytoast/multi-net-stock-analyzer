"""Minimal markdown table helpers (avoids an extra dependency)."""

from __future__ import annotations

import pandas as pd


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def md_frame(df: pd.DataFrame) -> str:
    return md_table([""] + [str(c) for c in df.columns], [[str(i), *row] for i, row in zip(df.index, df.to_numpy().tolist())])
