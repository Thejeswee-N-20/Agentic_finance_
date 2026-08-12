"""Trust the OS certificate store for outbound HTTPS (Windows TLS-scanning AV/proxies).

Some environments (e.g. Avast/AVG "Web Shield", corporate proxies) intercept HTTPS
and re-sign certificates with a locally-installed root CA. That root lives in the
Windows certificate store but **not** in certifi's bundle, so ``requests`` and
``curl_cffi`` (used by yfinance) fail with ``CERTIFICATE_VERIFY_FAILED``.

This module builds a combined CA bundle = certifi's CAs + the Windows ROOT/CA stores,
caches it next to the package, and points the standard cert env vars at it. It is a
no-op on non-Windows platforms and never disables verification. Existing env-var
overrides are respected (we only fill in unset ones).
"""
from __future__ import annotations

import os
import ssl
import sys

__all__ = ["ensure_ca_bundle"]

# Env vars read by the HTTP stacks we use: requests (REQUESTS_CA_BUNDLE),
# curl_cffi/libcurl (CURL_CA_BUNDLE), and the stdlib ssl default (SSL_CERT_FILE).
_CERT_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cacerts", "combined-ca.pem")


def _build_bundle(path: str) -> bool:
    """Write certifi's CAs plus the Windows cert stores to ``path``. Returns success."""
    try:
        import certifi
    except Exception:
        return False

    try:
        with open(certifi.where(), "rb") as fh:
            pem = bytearray(fh.read())
    except Exception:
        return False

    # Append every cert from the Windows ROOT and intermediate CA stores. Use the
    # stdlib DER->PEM converter (pure base64, no parsing) so malformed-but-trusted
    # certs from AV products don't trip a strict X.509 parser.
    added = 0
    for store in ("ROOT", "CA"):
        try:
            for der, _enc, _trust in ssl.enum_certificates(store):
                try:
                    pem += b"\n" + ssl.DER_cert_to_PEM_cert(der).encode("ascii")
                    added += 1
                except Exception:
                    continue
        except Exception:
            continue

    if added == 0:
        return False  # nothing to add; leave certifi's default in place

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(bytes(pem))
        return True
    except Exception:
        return False


# Rebuild the cached bundle after this many days: TLS-scanning AV products
# (e.g. Avast) rotate their local root periodically, and a stale bundle then
# fails verification with a misleading "no data" symptom downstream.
_MAX_AGE_DAYS = 7


def _is_stale(path: str) -> bool:
    try:
        import time
        return (time.time() - os.path.getmtime(path)) > _MAX_AGE_DAYS * 86400
    except OSError:
        return True


def ensure_ca_bundle() -> str | None:
    """On Windows, ensure a fresh combined CA bundle exists and export it via env vars.

    Returns the bundle path if active, else ``None``. Safe to call repeatedly.
    The cached bundle is rebuilt when older than ``_MAX_AGE_DAYS`` so a rotated
    AV/proxy root certificate cannot silently break live HTTPS.
    """
    if not sys.platform.startswith("win"):
        return None

    # If the user already pointed the stack at a bundle, don't second-guess them.
    if any(os.environ.get(v) for v in _CERT_ENV_VARS):
        return os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")

    path = _CACHE_PATH
    if not os.path.exists(path) or _is_stale(path):
        if not _build_bundle(path) and not os.path.exists(path):
            return None  # rebuild failed and no previous bundle to fall back on

    for var in _CERT_ENV_VARS:
        os.environ.setdefault(var, path)
    return path
