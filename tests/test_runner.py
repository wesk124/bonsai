import hashlib
import json
import socket
import threading

import pytest
import torch
import transformers

from bonsai_bench.continuation import prefill
from bonsai_bench.codec_backend import get_codec_backend
from bonsai_bench.runner import (MODEL_REVISION, digest, initialize_output, local_evaluate,
                                 parse_conditions, read_samples, remote_evaluate)
from bonsai_bench.transport import FramedSocket, TcpDiagnosticPayload
from bonsai_bench.worker import receive_request, serve_connection


class TinyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(x) for x in ids)

    def __call__(self, prompt, add_special_tokens=False):
        return {"input_ids": [int(x) for x in prompt.split()]}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(43)
    config = transformers.Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=96,
                                     num_hidden_layers=2, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=32,
                                     max_position_embeddings=128, attention_dropout=0,
                                     eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "eager"
    return transformers.Qwen3ForCausalLM(config).to(torch.bfloat16).eval()


@pytest.fixture
def sample():
    ids = [3, 7, 4, 8, 5]
    prompt = " ".join(map(str, ids))
    return {"sample_id": "sample-0", "task": "niah_single_2", "context_tokens": 4096,
            "split": "pilot", "seed": 71201, "model": "tiny", "model_revision": MODEL_REVISION,
            "input_ids": ids, "input_length": len(ids), "max_new_tokens": 4,
            "outputs": ["42"], "prompt": prompt,
            "hashes": {"input_ids_sha256": digest(ids),
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}}


def test_resume_requires_exact_provenance_and_keeps_failures(tmp_path):
    out = tmp_path / "rows.jsonl"
    config = {"conditions": [("local_roundtrip", "int4")], "revision": MODEL_REVISION}
    fp, completed = initialize_output(out, config)
    assert completed == set()
    out.write_text(json.dumps({"run_fingerprint": fp, "kind": "prediction", "row_key": "a"}) + "\n" +
                   json.dumps({"run_fingerprint": fp, "kind": "failure", "row_key": "b"}) + "\n")
    with pytest.raises(ValueError, match="retry-failures"):
        initialize_output(out, config)
    assert initialize_output(out, config, retry_failures=True)[1] == {"a"}
    with pytest.raises(ValueError, match="provenance differs"):
        initialize_output(out, {**config, "revision": "other"})
    assert '"kind": "failure"' in out.read_text()


def test_frozen_manifest_ids_hash_and_duplicates(tmp_path, sample):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(sample) + "\n")
    assert read_samples([path]) == [sample]
    with pytest.raises(ValueError, match="duplicate"):
        read_samples([path, path])
    sample["input_ids"][0] = 50
    path.write_text(json.dumps(sample) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_samples([path])


def test_condition_rejects_ambiguous_baseline():
    assert parse_conditions(["transfer:int4", "text_refill"]) == [("transfer", "int4"), ("text_refill", "bf16")]
    with pytest.raises(ValueError):
        parse_conditions(["text_refill:int4"])


def test_backend_selection_is_explicit_and_part_of_resume_provenance(tmp_path):
    from bonsai_bench import packed
    assert get_codec_backend() is packed
    with pytest.raises(ValueError, match="unknown codec backend"):
        get_codec_backend("automatic_fallback")
    out = tmp_path / "rows.jsonl"
    initialize_output(out, {"codec_backend": "pytorch_packed_v1"})
    with pytest.raises(ValueError, match="provenance differs"):
        initialize_output(out, {"codec_backend": "triton_packed_v1"})


def test_worker_rejects_backend_mismatch_before_allocating_or_transferring():
    sender, receiver = socket.socketpair()
    try:
        FramedSocket(sender).send_json({"kind": "hello", "model": "tiny", "revision": MODEL_REVISION,
                                       "transport": "tcp_diagnostic", "codec_backend": "triton_packed_v1"})
        with pytest.raises(ValueError, match="sender codec backend differs"):
            serve_connection(receiver, None, None, model_name="tiny", revision=MODEL_REVISION,
                             device="cpu", transport="tcp_diagnostic")
    finally:
        sender.close()
        receiver.close()
    with pytest.raises(ValueError, match="request codec backend differs"):
        receive_request(None, None, None, None, {"codec_backend": "triton_packed_v1"}, "cpu")


def test_local_actual_packed_roundtrips(model, sample):
    state = prefill(model, torch.tensor([sample["input_ids"]]))
    tokenizer = TinyTokenizer()
    raw = local_evaluate(model, tokenizer, state, sample, "local_bf16", "bf16", "cpu")
    decoded = local_evaluate(model, tokenizer, state, sample, "local_roundtrip", "bf16", "cpu")
    assert raw["generated_ids"] == decoded["generated_ids"]
    for precision in ("int8", "int4"):
        result = local_evaluate(model, tokenizer, state, sample, "local_roundtrip", precision, "cpu")
        assert result["payload_bytes"] > 0
        if precision == "int8":
            assert result["payload_bytes"] < decoded["payload_bytes"]
        assert result["generated_ids"][0] == raw["generated_ids"][0]
        assert result["wire_bytes"] > result["payload_bytes"]
        assert result["encode_ms"] > 0 and result["decode_ms"] > 0
    assert state.cache.get_seq_length() == sample["input_length"]


@pytest.mark.parametrize("condition,precision", [("transfer", "bf16"), ("transfer", "int8"),
                                                 ("transfer", "int4"), ("text_refill", "bf16")])
def test_two_worker_tcp_path_matches_local_and_counts_bytes(model, sample, condition, precision):
    tokenizer = TinyTokenizer()
    sender, receiver = socket.socketpair()
    errors = []

    def run():
        try:
            serve_connection(receiver, model, tokenizer, model_name="tiny", revision=MODEL_REVISION,
                             device="cpu", transport="tcp_diagnostic", timeout=30)
        except Exception as exc:
            errors.append(exc)
        finally:
            receiver.close()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        channel = FramedSocket(sender)
        channel.send_json({"kind": "hello", "model": "tiny", "revision": MODEL_REVISION,
                           "transport": "tcp_diagnostic"})
        assert channel.recv_json()["kind"] == "hello"
        state = prefill(model, torch.tensor([sample["input_ids"]]))
        result = remote_evaluate(channel, TcpDiagnosticPayload(), model, state, sample,
                                 condition, precision, "cpu", tag=1)
        local = local_evaluate(model, tokenizer, state, sample,
                               "local_bf16" if condition == "text_refill" else "local_roundtrip", precision, "cpu")
        assert result["generated_ids"] == local["generated_ids"]
        assert result["transport"] == "tcp_diagnostic"
        assert result["control_bytes"] > 0
        assert result["wire_bytes"] == result["payload_bytes"] + result["metadata_bytes"] + result["control_bytes"]
        assert result["prediction_response_bytes"] > 0
        assert result["handoff_ack_ms"] > 0
        assert result["receiver_memory"]["native_registered_bytes"] == 0
    finally:
        sender.close()
        thread.join(30)
    assert not thread.is_alive()
    assert not errors
