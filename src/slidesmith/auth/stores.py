"""Persistent stores for long-lived authentication session tokens."""

from __future__ import annotations

import contextlib
import json
import math
import os
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from slidesmith.engine.json_utils import read_json

try:
    import keyring as _keyring

    _KEYRING_AVAILABLE = True
except ImportError:
    _keyring = None  # type: ignore[assignment]
    _KEYRING_AVAILABLE = False

_KEYRING_SERVICE = "extrasuite"
_DEFAULT_PROFILE = "default"
_KEYRING_TIMEOUT_ENV = "SLIDESMITH_KEYRING_TIMEOUT"
_DEFAULT_NONINTERACTIVE_KEYRING_TIMEOUT = 5.0
_KEYRING_TIMEOUT_MESSAGE = (
    "keyring operation timed out; a macOS Keychain authorization prompt is likely "
    "pending"
)
_keyring_locks: dict[tuple[str, str], threading.Lock] = {}
_keyring_locks_guard = threading.Lock()
_invalid_timeout_warnings: set[str] = set()
_invalid_timeout_warnings_lock = threading.Lock()


class KeyringTimeoutError(RuntimeError):
    """An OS keyring operation exceeded its configured headless bound."""

    def __init__(self) -> None:
        super().__init__(_KEYRING_TIMEOUT_MESSAGE)


def _warn_invalid_keyring_timeout(value: str) -> None:
    with _invalid_timeout_warnings_lock:
        if value in _invalid_timeout_warnings:
            return
        _invalid_timeout_warnings.add(value)
    print(
        f"warning: ignoring invalid {_KEYRING_TIMEOUT_ENV}={value!r}; "
        "using the default keyring timeout",
        file=sys.stderr,
    )


def _keyring_timeout() -> float | None:
    """Resolve the per-call keyring bound, with interactive compatibility."""
    configured = os.environ.get(_KEYRING_TIMEOUT_ENV)
    if configured is not None:
        try:
            timeout = float(configured)
        except (TypeError, ValueError):
            _warn_invalid_keyring_timeout(configured)
        else:
            if math.isfinite(timeout):
                return timeout
            _warn_invalid_keyring_timeout(configured)

    try:
        interactive = bool(sys.stdin.isatty() and sys.stderr.isatty())
    except (AttributeError, OSError):
        interactive = False
    return None if interactive else _DEFAULT_NONINTERACTIVE_KEYRING_TIMEOUT


def _call_keyring(
    operation: Callable[[], Any], *, key: tuple[str, str] | None = None
) -> Any:
    """Run one keyring operation without unbounded headless blocking."""
    timeout = _keyring_timeout()
    if timeout is not None and timeout <= 0:
        raise KeyringTimeoutError()

    lock: threading.Lock | None = None
    deadline = time.monotonic() + timeout if timeout is not None else None
    if key is not None:
        with _keyring_locks_guard:
            lock = _keyring_locks.setdefault(key, threading.Lock())
        if deadline is None:
            lock.acquire()
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not lock.acquire(timeout=remaining):
                raise KeyringTimeoutError()

    result: list[Any] = []
    error: list[BaseException] = []

    def invoke() -> None:
        try:
            result.append(operation())
        except BaseException as exc:
            error.append(exc)
        finally:
            if lock is not None:
                lock.release()

    worker = threading.Thread(target=invoke, daemon=True)
    worker_owns_lock = False
    try:
        worker.start()
        worker_owns_lock = lock is not None
        if deadline is None:
            worker.join()
        else:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                worker.join(remaining)
        if worker.is_alive():
            raise KeyringTimeoutError()
        if error:
            raise error[0]
        return result[0]
    finally:
        if lock is not None and not worker_owns_lock:
            lock.release()


def _write_secure_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace a JSON file with mode 0600 from creation onward."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(stat.S_IRWXU)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd = os.open(
        temp_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            with contextlib.suppress(OSError):
                os.close(fd)
            raise
        with handle:
            handle.write(content)
        os.replace(temp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise


@dataclass
class SessionToken:
    """Stored authentication token for headless agent access."""

    raw_token: str | None = None
    email: str = ""
    expires_at: float = 0.0
    is_refreshable: bool = True
    access_token: str | None = None
    issued_at: float | None = None

    def __post_init__(self) -> None:
        # Accept the earlier in-progress access-only object shape in memory,
        # but never serialize it back under the legacy raw_token key.
        if not self.is_refreshable and self.access_token is None:
            self.access_token = self.raw_token

    def is_valid(self, buffer_seconds: int = 300) -> bool:
        """Check if session token is still valid with a 5-minute buffer."""
        return time.time() < self.expires_at - buffer_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        payload: dict[str, Any] = {
            "email": self.email,
            "expires_at": self.expires_at,
            "is_refreshable": self.is_refreshable,
        }
        if self.issued_at is not None:
            payload["issued_at"] = self.issued_at
        if self.is_refreshable:
            payload["raw_token"] = self.raw_token
        else:
            payload["access_token"] = self.access_token
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionToken:
        """Create SessionToken from dictionary."""
        if not isinstance(data, dict):
            raise TypeError("session token payload must be an object")
        issued_at = data.get("issued_at")
        try:
            if (
                isinstance(issued_at, bool)
                or not isinstance(issued_at, (int, float))
                or not math.isfinite(issued_at)
            ):
                issued_at = None
        except (OverflowError, ValueError):
            issued_at = None
        is_refreshable = data.get("is_refreshable", True)
        if not isinstance(is_refreshable, bool):
            raise TypeError("is_refreshable must be a boolean")
        if is_refreshable:
            return cls(
                raw_token=data["raw_token"],
                email=data["email"],
                expires_at=data["expires_at"],
                is_refreshable=True,
                issued_at=issued_at,
            )
        if "access_token" in data:
            return cls(
                access_token=data["access_token"],
                email=data["email"],
                expires_at=data["expires_at"],
                is_refreshable=False,
                issued_at=issued_at,
            )
        # Read the short-lived shape emitted by the earlier Phase 5 draft too.
        return cls(
            raw_token=data["raw_token"],
            email=data["email"],
            expires_at=data["expires_at"],
            is_refreshable=False,
            issued_at=issued_at,
        )


def _compare_sessions(a: SessionToken, b: SessionToken) -> int | None:
    """Compare session recency, returning ``None`` when it is ambiguous.

    Stamped tokens have exact issuance ordering. For legacy tokens, expiry is
    only a safe ordering signal within the same refreshability type; a
    cross-type pair with a missing stamp cannot be ordered safely.
    """
    if a.issued_at is not None and b.issued_at is not None:
        return (a.issued_at > b.issued_at) - (a.issued_at < b.issued_at)
    if a.is_refreshable == b.is_refreshable:
        return (a.expires_at > b.expires_at) - (a.expires_at < b.expires_at)
    return None


def _is_strictly_newer(candidate: SessionToken, existing: SessionToken) -> bool:
    """Return whether ``candidate`` is unambiguously newer than ``existing``."""
    comparison = _compare_sessions(candidate, existing)
    return comparison is not None and comparison > 0


class SessionStore(Protocol):
    """Protocol for session token storage backends."""

    def load(self, profile_name: str) -> SessionToken | None: ...
    def save(self, profile_name: str, token: SessionToken) -> None: ...
    def delete(self, profile_name: str) -> None: ...


class KeyringSessionStore:
    """Session token storage backed by the OS keyring."""

    @staticmethod
    def _backend() -> Any:
        # Preserve the historical monkeypatch surface on slidesmith.credentials.
        compat = sys.modules.get("slidesmith.credentials")
        available = getattr(compat, "_KEYRING_AVAILABLE", _KEYRING_AVAILABLE)
        backend = getattr(compat, "_keyring", _keyring)
        if not available or backend is None:
            raise RuntimeError("keyring package is not available")
        return backend

    def load(self, profile_name: str) -> SessionToken | None:
        raw = _call_keyring(
            lambda: self._backend().get_password(_KEYRING_SERVICE, profile_name),
            key=(_KEYRING_SERVICE, profile_name),
        )
        if not raw:
            return None
        try:
            token = SessionToken.from_dict(json.loads(raw))
            return token if token.is_valid() else None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Malformed-but-valid JSON (e.g. a list, or a non-numeric
            # expires_at) is treated as "no session", matching FileSessionStore.
            return None

    def save(self, profile_name: str, token: SessionToken) -> None:
        _call_keyring(
            lambda: self._backend().set_password(
                _KEYRING_SERVICE, profile_name, json.dumps(token.to_dict())
            ),
            key=(_KEYRING_SERVICE, profile_name),
        )

    def delete(self, profile_name: str) -> None:
        _call_keyring(
            lambda: self._backend().delete_password(_KEYRING_SERVICE, profile_name),
            key=(_KEYRING_SERVICE, profile_name),
        )


class FileSessionStore:
    """Session token storage at ``~/.config/slidesmith/session.json``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path)
            if path is not None
            else Path.home() / ".config" / "slidesmith" / "session.json"
        )

    def _load_profiles(self) -> dict[str, dict[str, Any]]:
        try:
            data = read_json(self.path, missing_ok=True)
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        profiles = data.get("profiles")
        if isinstance(profiles, dict):
            return {
                str(name): payload
                for name, payload in profiles.items()
                if isinstance(payload, dict)
            }
        if (
            {"email", "expires_at"}.issubset(data)
            and ("raw_token" in data or "access_token" in data)
        ):
            return {_DEFAULT_PROFILE: data}
        return {}

    def load(self, profile_name: str) -> SessionToken | None:
        payload = self._load_profiles().get(profile_name)
        if payload is None:
            return None
        try:
            token = SessionToken.from_dict(payload)
            return token if token.is_valid() else None
        except (KeyError, TypeError, ValueError):
            return None

    def save(self, profile_name: str, token: SessionToken) -> None:
        profiles = self._load_profiles()
        profiles[profile_name] = token.to_dict()
        _write_secure_json(self.path, {"profiles": profiles})

    def delete(self, profile_name: str) -> None:
        profiles = self._load_profiles()
        if profile_name not in profiles:
            return
        profiles.pop(profile_name)
        if profiles:
            _write_secure_json(self.path, {"profiles": profiles})
        else:
            self.path.unlink(missing_ok=True)


class FallbackSessionStore:
    """Read both persistent stores and return the newest valid session."""

    def __init__(
        self,
        keyring_store: SessionStore | None = None,
        file_store: SessionStore | None = None,
    ) -> None:
        self.keyring_store = keyring_store or KeyringSessionStore()
        self.file_store = file_store or FileSessionStore()
        self._notice_printed = False

    def _notice(self, exc: Exception) -> None:
        if self._notice_printed:
            return
        self._notice_printed = True
        print(
            f"warning: keyring unavailable ({exc!r}); using file session store",
            file=sys.stderr,
        )

    def load(self, profile_name: str) -> SessionToken | None:
        """Load a valid session, preferring access-only on ambiguous legacy pairs.

        Each store's ``load()`` already drops invalid tokens, so a surviving
        access-only token is provably less than about one hour old, while a
        refreshable token may be up to 30 days old. If a refreshable token is
        actually newer in that sub-hour race, both tokens still work; leaving
        the refreshable copy in its store lets ordering self-correct when the
        access-only token expires.
        """
        try:
            keyring_token = self.keyring_store.load(profile_name)
        except Exception as exc:
            self._notice(exc)
            return self.file_store.load(profile_name)
        file_token = self.file_store.load(profile_name)
        if keyring_token is None:
            return file_token
        if file_token is None:
            return keyring_token
        comparison = _compare_sessions(keyring_token, file_token)
        if comparison is None:
            return (
                keyring_token
                if not keyring_token.is_refreshable
                else file_token
            )
        return keyring_token if comparison >= 0 else file_token

    def save(self, profile_name: str, token: SessionToken) -> None:
        keyring_saved = False
        try:
            self.keyring_store.save(profile_name, token)
            keyring_saved = True
        except Exception as exc:
            self._notice(exc)
        try:
            self.file_store.save(profile_name, token)
        except Exception as exc:
            if not keyring_saved:
                raise
            print(
                f"warning: file session store unavailable ({exc!r}); "
                "session saved to keyring only",
                file=sys.stderr,
            )

    def delete(self, profile_name: str) -> None:
        try:
            self.keyring_store.delete(profile_name)
        except Exception as exc:
            self._notice(exc)
        self.file_store.delete(profile_name)


class InMemorySessionStore:
    """In-memory session token storage (for testing and non-persistent use)."""

    def __init__(self) -> None:
        self._tokens: dict[str, SessionToken] = {}

    def load(self, profile_name: str) -> SessionToken | None:
        token = self._tokens.get(profile_name)
        if token and token.is_valid():
            return token
        return None

    def save(self, profile_name: str, token: SessionToken) -> None:
        self._tokens[profile_name] = token

    def delete(self, profile_name: str) -> None:
        self._tokens.pop(profile_name, None)
