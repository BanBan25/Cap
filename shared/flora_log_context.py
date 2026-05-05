"""Optional logging context for nested jobs (e.g. rank sweep → mini federated run)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

_ctx: ContextVar[Optional[dict[str, int]]] = ContextVar("flora_sweep_ctx", default=None)


def set_sweep_context(sweep_client: int, sweep_rank: int) -> Token:
    """Call reset_sweep_context(token) when the sweep point finishes."""
    return _ctx.set({"client": int(sweep_client), "rank": int(sweep_rank)})


def reset_sweep_context(token: Token) -> None:
    _ctx.reset(token)


def log_prefix() -> str:
    d = _ctx.get()
    if not d:
        return ""
    return f"[sw c={d['client']} r={d['rank']}] "
