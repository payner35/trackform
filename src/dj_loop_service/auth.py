"""Bearer-token auth middleware (Phase 4a — single shared token).

If `DLP_API_TOKEN` env var is set, every request to `/v1/*` must carry
`Authorization: Bearer <token>` (or, for the WebSocket, `?token=<token>`).
If unset or empty, auth is disabled — used in local dev and tests.

Exempt paths:
  - `/v1/health` — must remain reachable for uptime checks.
  - `/v1/worker/*` — Caddy 404s these externally; only the in-cluster
    worker container hits them via the compose network. See
    DEPLOYMENT_SPEC §4.2.

Future (Phase 4b): worker traffic gets its own HMAC, separate from the
user token. Until then, defense-in-depth comes from Caddy not proxying
the worker path prefix.
"""

from __future__ import annotations

import hmac
import os
from fastapi import Request
from fastapi.responses import JSONResponse

_EXEMPT_PREFIXES = ("/v1/health", "/v1/worker/")


def _expected_token() -> str | None:
    tok = os.environ.get("DLP_API_TOKEN", "").strip()
    return tok or None


def _extract_bearer(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    parts = auth_header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


async def bearer_token_middleware(request: Request, call_next):
    expected = _expected_token()
    if expected is None:
        return await call_next(request)

    path = request.url.path
    if any(path == p or path.startswith(p) for p in _EXEMPT_PREFIXES):
        return await call_next(request)
    if not path.startswith("/v1/"):
        return await call_next(request)

    presented = _extract_bearer(request.headers.get("authorization"))
    if presented is None or not hmac.compare_digest(presented, expected):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    return await call_next(request)


def check_ws_token(token: str | None) -> bool:
    """Validate a WebSocket connection's `?token=` query param.

    Returns True if auth is disabled or the token matches; False otherwise.
    """
    expected = _expected_token()
    if expected is None:
        return True
    if token is None:
        return False
    return hmac.compare_digest(token, expected)
