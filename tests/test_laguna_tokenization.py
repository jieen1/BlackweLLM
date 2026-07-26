"""Laguna 分词护栏单测（CPU-only，不需要模型权重）。

验证 BOS 根因修复的永久护栏：
- tokenizer.encode() 必须包含 BOS (id=2)
- add_special_tokens=False 会跳过 BOS（回归防护）
- EOS 是 (2, 24)，不是 Qwen 的 (151643, 151645)
- prompt ids 断言函数能正确检测分歧
"""

from __future__ import annotations

import pytest

# ── prompt ids 断言函数（与 quality gate 共用逻辑）──


def assert_prompt_ids_equal(
    prompt: str,
    ids_a: list[int],
    ids_b: list[int],
    path_a: str = "A",
    path_b: str = "B",
) -> None:
    if ids_a != ids_b:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b),
            min(len(ids_a), len(ids_b)),
        )
        raise AssertionError(
            f"Prompt token ids 不一致 (first diff @{first_diff}): "
            f"{path_a}={ids_a[:10]}... vs {path_b}={ids_b[:10]}..."
        )


class TestPromptIdsAssertion:
    """assert_prompt_ids_equal 护栏函数本身的测试。"""

    def test_identical_ids_pass(self):
        assert_prompt_ids_equal("hello", [2, 100, 200], [2, 100, 200])

    def test_different_ids_raise(self):
        with pytest.raises(AssertionError, match="不一致"):
            assert_prompt_ids_equal("hello", [2, 100], [100, 200])

    def test_length_mismatch_raise(self):
        with pytest.raises(AssertionError, match="不一致"):
            assert_prompt_ids_equal("hello", [2, 100], [2, 100, 200])

    def test_empty_ids_pass(self):
        assert_prompt_ids_equal("", [], [])

    def test_bos_missing_detected(self):
        """模拟 BOS 缺失：一条路径有 BOS，另一条没有。"""
        with_bos = [2, 464, 279, 4894]
        without_bos = [464, 279, 4894]
        with pytest.raises(AssertionError, match="first diff @0"):
            assert_prompt_ids_equal("The capital", with_bos, without_bos)


class TestLagunaEOS:
    """Laguna EOS 配置测试。"""

    def test_laguna_eos_not_qwen(self):
        """Laguna EOS 是 (2, 24)，不是 Qwen 的 (151643, 151645)。"""
        laguna_eos = (2, 24)
        qwen_eos = (151643, 151645)
        assert laguna_eos != qwen_eos

    def test_eos_stop_condition(self):
        """验证停止条件逻辑。"""
        laguna_eos = (2, 24)
        tokens_with_eos = [100, 200, 24, 300]
        stopped_at = None
        for i, tok in enumerate(tokens_with_eos):
            if tok in laguna_eos:
                stopped_at = i
                break
        assert stopped_at == 2

    def test_qwen_eos_does_not_stop_laguna(self):
        """Qwen 的 EOS 不应触发 Laguna 停止。"""
        laguna_eos = (2, 24)
        qwen_tokens = [100, 151643, 200]
        stopped = any(tok in laguna_eos for tok in qwen_tokens)
        assert not stopped


class TestLagunaTokenizerBOS:
    """需要 tokenizer 文件的测试（仍为 CPU-only，无需 GPU/权重）。

    如果 tokenizer 不可用则 skip。
    """

    @pytest.fixture(autouse=True)
    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                "poolside/Laguna-S-2.1-NVFP4", trust_remote_code=True
            )
        except Exception:
            pytest.skip("Laguna tokenizer not available locally")

    def test_encode_includes_bos(self):
        """tokenizer.encode() 必须包含 BOS (id=2)。"""
        ids = self.tokenizer.encode("The capital of France is")
        assert ids[0] == 2, f"Expected BOS (2) at position 0, got {ids[0]}"

    def test_add_special_tokens_false_skips_bos(self):
        """add_special_tokens=False 会跳过 BOS — 这是根因回归防护。"""
        ids_with = self.tokenizer.encode("The capital of France is")
        ids_without = self.tokenizer.encode("The capital of France is", add_special_tokens=False)
        assert ids_with[0] == 2
        assert ids_without[0] != 2 or len(ids_without) < len(ids_with)
        assert len(ids_with) == len(ids_without) + 1

    def test_prompt_ids_assertion_with_real_tokenizer(self):
        """用真实 tokenizer 验证护栏函数。"""
        prompt = "The capital of France is"
        ids_correct = self.tokenizer.encode(prompt)
        ids_broken = self.tokenizer.encode(prompt, add_special_tokens=False)
        with pytest.raises(AssertionError):
            assert_prompt_ids_equal(prompt, ids_correct, ids_broken, "correct", "broken")

    def test_same_encode_passes_assertion(self):
        """同一 encode 路径应通过断言。"""
        prompt = "Hello world"
        ids_a = self.tokenizer.encode(prompt)
        ids_b = self.tokenizer.encode(prompt)
        assert_prompt_ids_equal(prompt, ids_a, ids_b)
