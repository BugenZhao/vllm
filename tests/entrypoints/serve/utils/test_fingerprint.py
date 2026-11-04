# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``system_fingerprint`` construction."""

from types import SimpleNamespace

import pytest

from vllm.entrypoints.serve.utils import fingerprint as fp

# The engine-reported config hash.
HASH = "a3b21f94deadbeef"


def _cfg(tp=1, pp=1, dp=1, ep=False):
    # No `compute_hash`: the fingerprint uses only the engine-reported hash.
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            enable_expert_parallel=ep,
        )
    )


@pytest.fixture(autouse=True)
def _reset():
    fp.set_default_fingerprint_mode("full")
    yield
    fp.set_default_fingerprint_mode("full")


def test_four_modes_produce_expected_shapes():
    from vllm import __version__ as v

    cfg = _cfg(tp=8, ep=True)

    assert fp.build_system_fingerprint(cfg, HASH, "full") == (
        f"vllm-{v}-tp8-ep-a3b21f94"
    )
    assert fp.build_system_fingerprint(cfg, HASH, "hash") == f"vllm-{v}-a3b21f94"
    assert fp.build_system_fingerprint(cfg, HASH, "custom", "my-fp") == "my-fp"
    assert fp.build_system_fingerprint(cfg, HASH, "none") is None


def test_full_mode_emits_only_non_trivial_parallelism():
    from vllm import __version__ as v

    # Single-GPU: nothing between version and hash.
    assert fp.build_system_fingerprint(_cfg(), HASH, "full") == f"vllm-{v}-a3b21f94"
    # All parallelism axes.
    assert (
        fp.build_system_fingerprint(_cfg(tp=8, pp=2, dp=4, ep=True), HASH, "full")
        == f"vllm-{v}-tp8-pp2-dp4-ep-a3b21f94"
    )


def test_get_respects_set_default():
    cfg = _cfg(tp=8)
    full = fp.get_system_fingerprint(cfg, HASH)
    assert full == fp.get_system_fingerprint(cfg, HASH)

    fp.set_default_fingerprint_mode("hash")
    hashed = fp.get_system_fingerprint(cfg, HASH)
    assert hashed is not None
    assert hashed != full
    assert "tp8" not in hashed

    fp.set_default_fingerprint_mode("custom", "deploy-42")
    assert fp.get_system_fingerprint(cfg, HASH) == "deploy-42"

    fp.set_default_fingerprint_mode("none")
    assert fp.get_system_fingerprint(cfg, HASH) is None


def test_missing_engine_hash_reports_nohash():
    full_fp = fp.build_system_fingerprint(_cfg(tp=2), None, "full")
    assert full_fp is not None and full_fp.endswith("-tp2-nohash")
    hash_fp = fp.build_system_fingerprint(_cfg(), None, "hash")
    assert hash_fp is not None and hash_fp.endswith("-nohash")
