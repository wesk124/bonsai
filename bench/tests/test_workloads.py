import numpy as np

from bonsai_bench.workloads import atom_lognormal, fixed


def test_atom_lognormal_is_deterministic_and_bounded():
    a, b = atom_lognormal(500, 2048), atom_lognormal(500, 2048)
    assert np.array_equal(a.prompt_lens, b.prompt_lens) and np.array_equal(a.output_lens, b.output_lens)
    assert a.prompt_lens.min() >= 1 and (a.prompt_lens + a.output_lens).max() <= 2048
    assert a.output_lens.min() >= 2
    # short prompts, long outputs: the regime Atom targets
    assert np.median(a.prompt_lens) < 64 < np.median(a.output_lens)


def test_fixed(tmp_path):
    rs = fixed([1024, 4096], output_len=16, per_len=3)
    assert list(rs.prompt_lens) == [1024] * 3 + [4096] * 3
    rs.save_jsonl(tmp_path / "w.jsonl")
    assert len((tmp_path / "w.jsonl").read_text().splitlines()) == 6
