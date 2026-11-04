# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the ``system_fingerprint`` string returned by the OpenAI-compatible
server.

Four modes, configured via ``--fingerprint-mode``:

* ``full`` (default): ``vllm-<version>[-<parallelism>]-<hash8>`` — encodes
  server version, any non-trivial parallelism degree (tp/pp/dp/ep), and an
  8-char prefix of the engine's config hash: ``compute_hash()`` of its
  initialized ``VllmConfig``, reported in its ready response (covers model
  identity, quant config, speculative, attention backend, etc.), or
  ``nohash`` when no engine reported one.
* ``hash``: ``vllm-<version>-<hash8>`` — parallelism stripped.
* ``custom``: user-provided literal via ``--fingerprint-value``.
* ``none``: the field is omitted (serialized as ``null``).

``get_system_fingerprint`` is only called at serving-class init (a handful
of times per server); each subclass caches the returned string on
``self.system_fingerprint``, so per-request cost is one attribute read.
"""

from __future__ import annotations

from typing import Any, Literal

FingerprintMode = Literal["full", "hash", "custom", "none"]

_DEFAULT_MODE: FingerprintMode = "full"
_CUSTOM_VALUE: str | None = None


def set_default_fingerprint_mode(
    mode: FingerprintMode,
    custom_value: str | None = None,
) -> None:
    """Configure the fingerprint mode for subsequent ``get_system_fingerprint``
    calls. Called once at server startup."""
    global _DEFAULT_MODE, _CUSTOM_VALUE
    _DEFAULT_MODE = mode
    _CUSTOM_VALUE = custom_value


def get_system_fingerprint(vllm_config: Any, config_hash: str | None) -> str | None:
    """Return the fingerprint using the mode configured by
    ``set_default_fingerprint_mode``. ``config_hash`` is the engine-reported
    config hash; ``vllm_config`` supplies the parallelism."""
    return build_system_fingerprint(
        vllm_config, config_hash, _DEFAULT_MODE, _CUSTOM_VALUE
    )


def build_system_fingerprint(
    vllm_config: Any,
    config_hash: str | None,
    mode: FingerprintMode = "full",
    custom_value: str | None = None,
) -> str | None:
    if mode == "none":
        return None
    if mode == "custom":
        return custom_value

    from vllm import __version__ as vllm_version

    hash8 = config_hash[:8] if config_hash else "nohash"

    if mode == "hash":
        return f"vllm-{vllm_version}-{hash8}"

    # mode == "full"
    parts: list[str] = [f"vllm-{vllm_version}"]
    pc = getattr(vllm_config, "parallel_config", None)
    if pc is not None:
        tp = getattr(pc, "tensor_parallel_size", 1)
        if tp > 1:
            parts.append(f"tp{tp}")
        pp = getattr(pc, "pipeline_parallel_size", 1)
        if pp > 1:
            parts.append(f"pp{pp}")
        dp = getattr(pc, "data_parallel_size", 1)
        if dp > 1:
            parts.append(f"dp{dp}")
        if getattr(pc, "enable_expert_parallel", False):
            parts.append("ep")
    parts.append(hash8)
    return "-".join(parts)
