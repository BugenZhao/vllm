# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import msgspec
import numpy as np
import pytest
import torch.cuda

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore, _add_request_ids
from vllm.v1.serial_utils import MsgpackEncoder

MODEL_NAME = "vllm-project/tiny-random-LlamaForCausalLM"


def test_preprocess_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Test that preprocessing errors are handled gracefully."""
    if current_platform.is_xpu():
        pytest.skip(
            "Skipped on XPU: this test only works with 'fork', but XPU uses 'spawn'."
        )

    assert not torch.cuda.is_initialized(), (
        "fork needs to be used for the engine "
        "core process and this isn't possible if cuda is already initialized"
    )

    # Store original method to call for non-failing requests
    original_preprocess = EngineCore.preprocess_add_request

    # Monkeypatch to make preprocess_add_request raise an exception
    # only for requests with "FAIL" in the first token
    def conditional_failing_preprocess(self, request: EngineCoreRequest):
        # Fail if the first token id is 333
        if request.prompt_token_ids is not None and request.prompt_token_ids[0] == 333:
            raise ValueError("Simulated preprocessing error!")
        return original_preprocess(self, request)

    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", conditional_failing_preprocess
    )
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    llm = LLM(model=MODEL_NAME)

    # Create a failing request by crafting a request with an invalid token
    # We need to use a direct approach since LLM.generate tokenizes for us
    # This should raise an exception due to the preprocessing failure
    # Special token id to trigger the failure
    failing_prompt = TokensPrompt(prompt_token_ids=[333])
    outputs = llm.generate(failing_prompt, SamplingParams(max_tokens=10))  # type: ignore
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 0
    assert outputs[0].finished
    assert outputs[0].outputs[0].finish_reason == "error"

    # Verify the engine is still functional with a normal request
    outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
    assert outputs[0].outputs[0].finish_reason in ("stop", "length")


def test_engine_decode_validation_error(tmp_path):
    """A request that passes the frontend's checks but fails them when the
    engine decodes it fails alone, and the engine keeps taking requests.

    `SamplingParams.__post_init__` runs again on decode, after the frontend has
    merged the generation config's EOS ids into `stop_token_ids`: 128 ids pass
    the frontend's `VLLM_MAX_STOP_TOKEN_IDS` check, 130 reach the engine.
    """
    # Two extra EOS ids, as in Llama 3's generation config.
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [1000, 1001]})
    )
    llm = LLM(model=MODEL_NAME, generation_config=str(tmp_path))

    params = SamplingParams(max_tokens=10, stop_token_ids=list(range(128)))
    [output] = llm.generate(TokensPrompt(prompt_token_ids=[1, 2, 3]), params)
    assert output.finished
    assert len(output.outputs[0].token_ids) == 0
    assert output.outputs[0].finish_reason == "error"

    outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0


@pytest.mark.parametrize(
    "fields,expected",
    [
        (
            msgspec.msgpack.decode(
                MsgpackEncoder().encode(
                    EngineCoreRequest(
                        request_id="req-0",
                        prompt_token_ids=np.array([1, 2, 3], dtype=np.int32),
                        mm_features=None,
                        sampling_params=SamplingParams(),
                        pooling_params=None,
                        arrival_time=0.0,
                        lora_request=None,
                        cache_salt=None,
                        data_parallel_rank=None,
                        client_index=3,
                    )
                )[0]
            ),
            ("req-0", 3),
        ),
        # An encoder may leave out trailing default fields, client_index included.
        (["req-0"], ("req-0", 0)),
        ([], None),
        ([1], None),  # a non-string request id
        (["req-0"] + [None] * 10 + [True], None),  # a non-int client index
        (None, None),  # the frame was not msgpack
    ],
)
def test_add_request_ids(fields, expected):
    """The input thread recovers the ids it needs to fail a request whose
    typed decode raised, and gives up on arrays that do not hold them."""
    assert _add_request_ids(fields) == expected
